import torch
import torchvision
import numpy as np
import PIL
import collections 
import random
import cv2
import os

__all__ = ['VideoFilePathToTensor', 'VideoFolderPathToTensor', 'VideoResize', 'VideoRandomCrop', 'VideoCenterCrop', 'VideoRandomHorizontalFlip', 
            'VideoRandomVerticalFlip', 'VideoGrayscale']


class VideoFolderPathToTensor(object):
  

    def __init__(self, max_len=None, padding_mode=None):
        self.max_len = max_len
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode

    def __call__(self, path):
        """
        Args:
            path (str): path of video folder.
            
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
       
        try:
            # get video properity
            timestamp = [i for i in range(0, 90, 3)]
            frames_path =([os.path.join(path, '%s.png') % timestamps for timestamps in sorted(timestamp, key=int) ])
            #frames_path = sorted([os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            frame = cv2.imread(frames_path[0])
            height = frame.shape[0]
            width= frame.shape[1]
            channels= frame.shape[2]
            num_frames = len(frames_path)
        except IndexError:
            # Handle the case where the folder has fewer timestamps
            print("Folder has fewer timestamps, skipping...")


        # init empty output frames (C x L x H x W)
        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(num_frames, self.max_len)
        else:
            # time length is unlimited
            time_len = num_frames

        frames = torch.FloatTensor(channels, time_len, height, width)

        # load the video to tensor
        for index in range(time_len):
            if index < num_frames:
                # frame exists
                # read frame
                frame = cv2.imread(frames_path[index])
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert(index > 0)
                    frames[:, index:, :, :] = frames[:, index-1, :, :].view(channels, 1, height, width)
                break

        frames /= 255
        return frames

class VideoResize(object):
    """ resize video tensor (C x L x H x W) to (C x L x h x w) 
    
    Args:
        size (sequence): Desired output size. size is a sequence like (H, W),
            output size will matched to this.
        interpolation (int, optional): Desired interpolation. Default is 'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, collections.abc.Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to be scaled (C x L x H x W)
        
        Returns:
            torch.Tensor: Rescaled video (C x L x h x w)
        """

        h, w = self.size
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)
        
        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

        
class VideoRandomCrop(object):
    """ Crop the given Video Tensor (C x L x H x W) at a random location.

    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size
    
    def __call__(self, video):
        """ 
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top : top + h, left : left + w]
        
        return video
        
class VideoCenterCrop(object):
    """ Crops the given video tensor (C x L x H x W) at the center.

    Args:
        size (sequence): Desired output size of the crop like (h, w).
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.
        
        Returns:
            torch.Tensor: Cropped Video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w
        
        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top : top + h, left : left + w]
        
        return video
        
class VideoRandomHorizontalFlip(object):
    """ Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video
            

class VideoRandomVerticalFlip(object):
    """ Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video
            
class VideoGrayscale(object):
    """ Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output video
    """

    def __init__(self, num_output_channels=1):
        assert num_output_channels in (1, 3)
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (3 x L x H x W) to be converted to grayscale.

        Returns:
            torch.Tensor: Grayscaled video (1 x L x H x W  or  3 x L x H x W)
        """

        C, L, H, W = video.size()
        grayscaled_video = torch.FloatTensor(self.num_output_channels, L, H, W)
        
        # use torchvision implemention to convert video frames to gray scale
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(self.num_output_channels),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            grayscaled_video[:, l, :, :] = frame

        return grayscaled_video

class One_hot_encoding(object):
    
    def __call__(self, label):


        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.label)
        print(integer_encoded)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        
        return onehot_encoded



        
