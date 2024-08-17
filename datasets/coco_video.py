from collections import defaultdict

from pycocotools.coco import COCO


class COCOVideo(COCO):
    def __init__(self, annotation_file=None):
        super(COCOVideo, self).__init__(annotation_file)

        self.videos = dict()
        self.vidToImgs = defaultdict(list)

        if not annotation_file == None:
            print('loading video annotations into memory...')
            self.createVideoIndex()
            print('Done')

    def createVideoIndex(self):
        videos = {}
        vidToImgs = defaultdict(list)  # vidToImgs[video_id] gives all image_id's of that video (as a list)

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                videos[video['id']] = video

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                vidToImgs[img['video_id']].append(img['id'])

        # create class members
        self.vidToImgs = vidToImgs
        self.videos = videos
