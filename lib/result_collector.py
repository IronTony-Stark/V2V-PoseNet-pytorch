import numpy as np

class BatchResultCollector():
    def __init__(self, samples_num, transform_output):
        self.samples_num = samples_num
        self.transform_output = transform_output
        self.pred_keypoints = None
        self.gt_keypoints = None
        self.idx = 0

    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.detach().cpu().numpy()
        refpoints_batch, joints_batch = extra_batch['refpoints'].cpu().numpy(), extra_batch['joints'].cpu().numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.pred_keypoints is None:
            # Initialize keypoints until dimensions available now
            self.pred_keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))
            self.gt_keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0]
        self.pred_keypoints[self.idx:self.idx + batch_size] = keypoints_batch
        self.gt_keypoints[self.idx:self.idx + batch_size] = joints_batch
        self.idx += batch_size

    def get_pred_keypoints(self):
        return self.pred_keypoints

    def get_gt_keypoints(self):
        return self.gt_keypoints

    def reset(self):
        self.pred_keypoints = None
        self.gt_keypoints = None
        self.idx = 0
