import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='ILVIT results refactoring script')
    parser.add_argument(
        '--log_file', type=str, required=True, help='Log file.')

    # get command line arguments
    args = parser.parse_args()
    log_file = args.log_file
    folder = os.path.dirname(log_file)

    print('Reading file', log_file)

    epochs_log = []
    for line in open(log_file, 'r'):
        epochs_log.append(json.loads(line))

    # print metrics from last epoch
    detection_metrics =list(epochs_log[-1]['val_metrics_coco_eval_bbox'].values())
    mot_metrics = [val for pair in zip(list(epochs_log[-1]['val_metrics_mota'].values()), list(epochs_log[-1]['val_metrics_motp'].values())) for val in pair]
    excel_table_metrics = detection_metrics + mot_metrics
    excel_table_metrics = [round(n, 3) for n in excel_table_metrics]

    print(excel_table_metrics)

    # epochs = []
    # train_losses = []
    # test_losses = []
    # pycocotoolsmetrics = []
    # pycocotoolslegend = [
    #     '(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    #     '(AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    #     '(AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    #     '(AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    #     '(AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    #     '(AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    #     '(AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    # ]
    #
    # for e in epochs_log:
    #     epochs.append(e['epoch'])
    #     train_losses.append(e['train_loss'])
    #     test_losses.append(e['test_loss'])
    #     pycocotoolsmetrics.append(e['test_coco_eval_bbox'])
    #
    # # training loss
    # plt.plot(epochs, train_losses, label="Train")
    # plt.plot(epochs, test_losses, label="Test")
    # plt.legend(loc="upper left")
    # plt.savefig(os.path.join(folder, 'train_loss.png'))
    # plt.close()
    #
    # # pycocotools metrics
    # pycocotoolsmetrics = np.array(pycocotoolsmetrics)
    # #print(pycocotoolsmetrics)
    # #print(pycocotoolsmetrics.shape)
    # for m in range(len(pycocotoolslegend)):
    #     plt.plot(epochs, pycocotoolsmetrics[:,m], label=pycocotoolslegend[m])
    # plt.legend(loc="upper left", prop={'size': 6})
    # plt.savefig(os.path.join(folder, 'pycocotools_metrics.png'))
    # plt.close()

if __name__ == "__main__":
    main()