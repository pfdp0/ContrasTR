# from https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py

import os
import numpy as np
import json
import argparse

# Use the same script for MOT16
SPLITS = ['train_half', 'val_half', 'train', 'test']
OUT_SPLITS = ['train', 'val', 'train_full', 'test']
FILE_NAMES = ['train_half', 'val_half', 'train_full', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
VISIBILITY_THRESHOLD = 0.0


def main(args):
    for s, split in enumerate(SPLITS):
        data_path = args.data_path + (split if (not HALF_VIDEO or split == 'test') else 'train')
        out_dir = os.path.join('data/MOT17', OUT_SPLITS[s], 'annotations')
        os.makedirs(os.path.join(out_dir), exist_ok=True)
        out_path = os.path.join(out_dir, 'instances_{}.json'.format(FILE_NAMES[s]))
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}],
               'videos': []}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        print(seqs)
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            if 'MOT17' in args.data_path and (not ('FRCNN' in seq)):
                continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            img_path = seq_path + 'img1/'
            ann_path = seq_path + 'gt/gt.txt'
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                    [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
            for i in range(num_images):
                if (i < image_range[0] or i > image_range[1]):
                    continue
                image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                              'id': image_cnt + i + 1,
                              'frame_id': i + 1 - image_range[0],
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            if split != 'test':
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0]) if \
                                         int(anns[i][0]) - 1 >= image_range[0] and \
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    # gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
                    seq_out_path = os.path.join('data/MOT17', 'mot17_gt_eval', OUT_SPLITS[s], seq)
                    gt_out_folder = os.path.join(seq_out_path, 'gt')
                    os.makedirs(os.path.join(gt_out_folder), exist_ok=True)
                    gt_out = os.path.join(gt_out_folder, 'gt.txt')
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write(
                            '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                int(o[6]), int(o[7]), o[8]))
                    fout.close()
                    # write seqinfo.ini
                    seq_info_file = []
                    seq_info_file.extend(['[Sequence]'])
                    seq_info_file.extend(['name=' + seq])
                    seq_info_file.extend(['seqLength=' + str(image_range[1] - image_range[0] + 1)])

                    # write to file predictions of current sequence
                    with open(os.path.join(seq_out_path, "seqinfo.ini"), 'w') as file:
                        for line in seq_info_file:
                            file.write("%s\n" % line)

                print(' {} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    if not ('15' in args.data_path):
                        if not (float(anns[i][8]) >= VISIBILITY_THRESHOLD):
                            continue
                        if not (int(anns[i][6]) == 1):
                            continue
                        if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]):  # Non-person
                            continue
                        if (int(anns[i][7]) in [2, 7, 8, 12]):  # Ignored person
                            category_id = -1
                        else:
                            category_id = 1
                    else:
                        category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'instance_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': (anns[i][4] * anns[i][5]).item()}
                    out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))

    if CREATE_SPLITTED_ANN:
        seqmap_file = []
        seqmap_file.extend(['name'])
        seqmap_file.extend(['MOT17-02-FRCNN'])
        seqmap_file.extend(['MOT17-04-FRCNN'])
        seqmap_file.extend(['MOT17-05-FRCNN'])
        seqmap_file.extend(['MOT17-09-FRCNN'])
        seqmap_file.extend(['MOT17-10-FRCNN'])
        seqmap_file.extend(['MOT17-11-FRCNN'])
        seqmap_file.extend(['MOT17-13-FRCNN'])

        # write to file predictions of current sequence
        with open(os.path.join('data/MOT17', 'mot17_gt_eval', 'seqmaps', "MOT17-train.txt"), 'w') as file:
            for line in seqmap_file:
                file.write("%s\n" % line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MOT17 annotations to COCO format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, help=" MOT17 path.", required=True)
    args = parser.parse_args()

    main(args)