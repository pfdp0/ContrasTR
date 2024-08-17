import os
import json
import argparse


def main(args):
    out_dir = os.path.join('data/bdd100k_mot/test/annotations')
    os.makedirs(os.path.join(out_dir), exist_ok=True)
    out_path = os.path.join(out_dir, 'instances_test.json')
    out = {'images': [], 'annotations': [],
           'categories': [
        {
            "id": 1,
            "name": "pedestrian"
        },
        {
            "id": 2,
            "name": "rider"
        },
        {
            "id": 3,
            "name": "car"
        },
        {
            "id": 4,
            "name": "truck"
        },
        {
            "id": 5,
            "name": "bus"
        },
        {
            "id": 6,
            "name": "train"
        },
        {
            "id": 7,
            "name": "motorcycle"
        },
        {
            "id": 8,
            "name": "bicycle"
        }
    ],
           'videos': []}
    seqs = os.listdir(args.data_path)
    image_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
        print(seq)
        video_cnt += 1
        out['videos'].append({
            'id': video_cnt,
            'name': seq})
        seq_path = os.path.join(args.data_path, seq)
        images = os.listdir(seq_path)
        images.sort()
        num_images = len([image for image in images if 'jpg' in image])
        for i in range(num_images):
            image_info = {'file_name': '{}/{}'.format(seq, images[i]),
                          'id': image_cnt + i + 1,
                          'frame_id': i + 1,
                          'prev_image_id': image_cnt + i if i > 0 else -1,
                          'next_image_id': \
                              image_cnt + i + 2 if i < num_images - 1 else -1,
                          'video_id': video_cnt}
            out['images'].append(image_info)
        print('{}: {} images'.format(seq, num_images))
        image_cnt += num_images
    json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates BDD100K mot annotations to COCO format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, help=" BDD100K test images path.", required=True)
    args = parser.parse_args()

    main(args)