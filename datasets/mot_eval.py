import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import motmetrics as mm
import torch
import trackeval
from bdd100k.eval import run

from util.misc import reduce_dict, get_world_size

from util.box_ops import box_xyxy_to_xywh

METRICS_LIST = ['num_matches', 'num_false_positives', 'num_misses', 'num_switches', 'mota', 'motp', 'num_objects']


class MOTEvaluator(object):
    SCORE_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, objectness_threshold, iou_threshold: float = 0.5):

        self.score_thresholds = [t for t in MOTEvaluator.SCORE_THRESHOLDS if t >= objectness_threshold]

        self.accumulators = {
            score: defaultdict(mm.MOTAccumulator) for score in self.score_thresholds
        }
        self.sub_metrics_sum = {score: None for score in self.score_thresholds}

        self.already_processed = dict()

        self.mota = {str(score): None for score in self.score_thresholds}
        self.motp = {str(score): None for score in self.score_thresholds}

        self.mh = mm.metrics.create()

        self.iou_threshold = iou_threshold

    def update(self, video_ids, image_ids, current_frames, results, targets):
        """
        args:
            video_ids:
            image_ids:
            current_frames:
            results:
            targets:
        """
        for i, (video_id, image_id, current_frame) in enumerate(zip(video_ids, image_ids, current_frames)):
            if (video_id.item(), image_id) not in self.already_processed:  # if we didn't process it yet
                pred_boxes = results[i]["boxes"]
                pred_ids = results[i]["ids"]
                pred_scores = results[i]["scores"]
                target_boxes = targets[i]["boxes"].tolist()
                target_ids = targets[i]["instance_id"].tolist()

                for score_threshold in self.score_thresholds:
                    # mask predicted boxes that are unde the threshold
                    score_mask = (pred_scores >= score_threshold)
                    kept_pred_boxes = pred_boxes[score_mask].tolist()
                    kept_pred_ids = pred_ids[score_mask].tolist()

                    # compute IoU matrix and add to the accumulator
                    iou_matrix = mm.distances.iou_matrix(target_boxes, kept_pred_boxes, max_iou=self.iou_threshold)
                    self.accumulators[score_threshold][video_id.item()].update(target_ids, kept_pred_ids, iou_matrix,
                                                                               frameid=current_frame.item())

                self.already_processed[(video_id.item(), image_id)] = True

    def compute(self):
        for score_threshold in self.score_thresholds:
            summary = self.mh.compute_many(self.accumulators[score_threshold].values(),
                                           metrics=METRICS_LIST,
                                           names=self.accumulators[score_threshold].keys())

            summary['distances'] = summary['motp'] * summary['num_matches']

            print(summary)

            sub_metrics_sum = summary.sum().to_dict()  # get a dict
            self.sub_metrics_sum[score_threshold] = {k: torch.tensor(v).to('cuda') for k, v in sub_metrics_sum.items()}

    def synchronize_between_processes(self):
        world_size = get_world_size()
        if world_size > 1:
            for score_threshold in self.score_thresholds:
                self.sub_metrics_sum[score_threshold] = reduce_dict(self.sub_metrics_sum[score_threshold],
                                                                    average=False)

    def summarize(self):
        for score_threshold in self.score_thresholds:
            print(f"MOT sub-metrics @[ IoU={self.iou_threshold} and score_thrs={score_threshold} ]:")
            for k, v in self.sub_metrics_sum[score_threshold].items():
                print(f"\t{k} = {v}")

        for score_threshold in self.score_thresholds:
            self.mota[str(score_threshold)] = 1. - ((self.sub_metrics_sum[score_threshold]['num_misses'] +
                                                     self.sub_metrics_sum[score_threshold]['num_false_positives'] +
                                                     self.sub_metrics_sum[score_threshold]['num_switches']) /
                                                    self.sub_metrics_sum[score_threshold]['num_objects'])
            self.motp[str(score_threshold)] = self.sub_metrics_sum[score_threshold]['distances'] / \
                                              self.sub_metrics_sum[score_threshold]['num_matches']

            print(f"Evaluation results for MOT: "
                  f"\n Multiple Object Accuracy (MOTA)  @[ IoU={self.iou_threshold} and score_thrs={score_threshold} ] = {self.mota[str(score_threshold)]}"
                  f"\n Multiple Object Precision (MOTP) @[ IoU={self.iou_threshold} and score_thrs={score_threshold} ] = {self.motp[str(score_threshold)]}")


class MOTOfficialEvaluator(ABC):
    SCORE_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, labels_to_cats, objectness_threshold, data_path):

        self.score_thresholds = [t for t in MOTEvaluator.SCORE_THRESHOLDS if t >= objectness_threshold]
        self.predictions = {t: {} for t in self.score_thresholds}
        self.labels_to_cats = labels_to_cats
        self.data_path = data_path

    def add_predictions(self, video_names, current_frames, name_frames, results):
        # looping over batch size
        for i, (video_name, current_frame, name_frame) in enumerate(zip(video_names, current_frames, name_frames)):

            current_frame = current_frame.item()

            pred_boxes = results[i]["boxes"]
            pred_ids = results[i]["ids"]
            pred_scores = results[i]["scores"]
            pred_labels = results[i]["labels"]

            for score_threshold in self.score_thresholds:
                if video_name not in self.predictions[score_threshold]:
                    self.predictions[score_threshold][video_name] = {}
                if current_frame not in self.predictions[score_threshold][video_name]:
                    self.predictions[score_threshold][video_name][current_frame] = {}

                    # mask predicted boxes that are under the threshold
                    score_mask = (pred_scores >= score_threshold)
                    kept_pred_boxes = pred_boxes[score_mask].tolist()
                    kept_pred_ids = pred_ids[score_mask].tolist()
                    kept_scores = pred_scores[score_mask].tolist()
                    kept_labels = pred_labels[score_mask].tolist()
                    kept_cats = [self.labels_to_cats[l]['name'] for l in kept_labels]

                    frame_info = {'frame_name': name_frame}

                    frame_predictions = {'pred_boxes': kept_pred_boxes,
                                         'pred_ids': kept_pred_ids,
                                         'pred_scores': kept_scores,
                                         'pred_cats': kept_cats}

                    self.predictions[score_threshold][video_name][current_frame] = {'frame_info': frame_info,
                                                                                    'frame_predictions': frame_predictions}

    @abstractmethod
    def store_subprocess_predictions(self):
        pass

    @abstractmethod
    def merge_from_processes_predictions(self):
        pass

    @abstractmethod
    def evaluate_with_official_code(self):
        pass

class MOTOfficialEvaluatorBDD100K(MOTOfficialEvaluator):

    def __init__(self, labels_to_cats, objectness_threshold, data_path, out_dir):
        super().__init__(labels_to_cats, objectness_threshold, data_path)
        self.out_dir = out_dir

    def store_subprocess_predictions(self):

        for score_threshold in self.score_thresholds:
            score_folder = os.path.join(self.out_dir, "submission_files", "trackscore_" + str(score_threshold), 'data')
            os.makedirs(os.path.join(score_folder), exist_ok=True)
            for video_name in self.predictions[score_threshold].keys():
                tracking_results = []
                for f, frame_n in enumerate(self.predictions[score_threshold][video_name].keys()):
                    frame = self.predictions[score_threshold][video_name][frame_n]

                    frame_name = frame['frame_info']['frame_name']

                    frame_dict = {
                        "videoName": video_name,
                        "name": frame_name,
                        "frameIndex": f,
                        "labels": []
                    }

                    if len(frame['frame_predictions']['pred_boxes']) > 0:
                        for box, id, score, cat in zip(*frame['frame_predictions'].values()):

                            frame_dict["labels"].append({
                                "id": str(id),
                                "category": cat,
                                "score": score,
                                "attributes": {
                                    "crowd": False
                                },
                                "box2d": {"x1":box[0], "y1":box[1], "x2":box[2], "y2":box[3]}
                            })
                    tracking_results.extend([frame_dict])

                # write to file predictions of current sequence
                with open(os.path.join(score_folder, video_name + ".json"), 'w') as file:
                    json.dump(tracking_results, file)

    def merge_from_processes_predictions(self):
        pass

    def evaluate_with_official_code(self):

        for score_threshold in self.score_thresholds:

            print("Evaluation with official submission format and code. Score threshold: ", score_threshold)

            score_folder = os.path.join(self.out_dir, "submission_files", "trackscore_" + str(score_threshold), 'data')
            shutil.make_archive(score_folder, 'zip', score_folder)

            # python -m bdd100k.eval.run -t box_track -g ${gt_file} -r ${res_file}

            sys.argv = ['', '-t', 'box_track', '-g', os.path.join(self.data_path, 'bdd100k_gt_eval/val'), '-r', score_folder + '']
            run.run()

        print('Done')

        print("Evaluation with TrackEval")

        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = True
        default_eval_config['TIME_PROGRESS'] = False
        default_dataset_config = trackeval.datasets.BDD100K.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

        config['GT_FOLDER'] = os.path.join(self.data_path, "bdd100k_gt_eval", "val")
        config['TRACKERS_FOLDER'] = os.path.join(self.out_dir, "submission_files")
        config['SPLIT_TO_EVAL'] = 'val'

        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.BDD100K(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric())
        evaluator.evaluate(dataset_list, metrics_list)

        print('Done')



class MOTOfficialEvaluatorMOT17(MOTOfficialEvaluator):
    def __init__(self, labels_to_cats, objectness_threshold, data_path, out_dir):
        super().__init__(labels_to_cats, objectness_threshold, data_path)
        self.out_dir = out_dir

    def store_subprocess_predictions(self):

        for score_threshold in self.score_thresholds:
            score_folder = os.path.join(self.out_dir, "submission_files", "trackscore_" + str(score_threshold), 'data')
            os.makedirs(os.path.join(score_folder), exist_ok=True)
            for video_name in self.predictions[score_threshold].keys():
                instances_results = {}
                tracking_results = []
                for frame_n in self.predictions[score_threshold][video_name].keys():
                    frame = self.predictions[score_threshold][video_name][frame_n]
                    if len(frame['frame_predictions']['pred_boxes']) > 0:
                        for box, id, score, cat in zip(*frame['frame_predictions'].values()):
                            if id not in instances_results:
                                instances_results[id] = []
                            instances_results[id].append({"frame": frame_n, "id": id, "bbox": box, "conf": score})

                for id in instances_results.keys():
                    for e in instances_results[id]:

                        line_entry = []

                        line_entry.append(str(e["frame"] + 1))

                        line_entry.append(str(e["id"]))

                        line_entry.append(f'{e["bbox"][0]:.3f}')
                        line_entry.append(f'{e["bbox"][1]:.3f}')
                        line_entry.append(f'{e["bbox"][2]:.3f}')
                        line_entry.append(f'{e["bbox"][3]:.3f}')

                        line_entry.append(str(-1))

                        line_entry.append(str(-1))
                        line_entry.append(str(-1))
                        line_entry.append(str(-1))

                        tracking_results.append(line_entry)

                # write to file predictions of current sequence
                with open(os.path.join(score_folder, video_name + ".txt"), 'w') as file:
                    for line in tracking_results:
                        line = ",".join(line)
                        file.write("%s\n" % line)

    def merge_from_processes_predictions(self):
        pass

    def evaluate_with_official_code(self):

        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

        config['USE_PARALLEL'] = True
        config['TIME_PROGRESS'] = False
        config['GT_FOLDER'] = os.path.join(self.data_path, "mot17_gt_eval", "val")
        config['TRACKERS_FOLDER'] = os.path.join(self.out_dir, "submission_files")
        config['SEQMAP_FOLDER'] = os.path.join(self.data_path, "mot17_gt_eval", "seqmaps")
        config['SPLIT_TO_EVAL'] = 'train'
        config['SKIP_SPLIT_FOL'] = True
        config['LOG_ON_ERROR'] = os.path.join(self.out_dir, "submission_files", 'error_log.txt')

        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric())
        evaluator.evaluate(dataset_list, metrics_list)

        print('Done')


def build_official_mot_eval(dataset_file, data_path, objectness_threshold, output_folder, labels_to_cats):
    if dataset_file == 'bdd100k':
        mot_official_evaluator = MOTOfficialEvaluatorBDD100K(labels_to_cats, objectness_threshold, data_path, output_folder)
    elif dataset_file == 'mot17':
        mot_official_evaluator = MOTOfficialEvaluatorMOT17(labels_to_cats, objectness_threshold, data_path, output_folder)
    else:
        raise ValueError(f'unknown dataset {dataset_file}')

    return mot_official_evaluator
