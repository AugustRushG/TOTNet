from .losses import HeatmapBallDetectionLoss2DWeighted, Heatmap_Ball_Detection_Loss_Weighted
from .metrics import heatmap2d_calculate_metrics, precision_recall_f1_tracknet, extract_coords2d, calculate_rmse, heatmap_calculate_metrics, extract_coords, pck_calculation

__all__ = [
    'HeatmapBallDetectionLoss2DWeighted',
    'heatmap2d_calculate_metrics',
    'precision_recall_f1_tracknet',
    'extract_coords'
]



class Losses:
    def __init__(self, configs, loss_type='WBCE', device='cpu'):
        self.H = configs.img_size[0]
        self.W = configs.img_size[1]
        if loss_type == 'WBCE':
            print("Using WBCE for loss function")
            self.loss = HeatmapBallDetectionLoss2DWeighted(H=self.H, W=self.W).to(device)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        

    def compute_loss(self, output, target, visibility=None):
        return self.loss(output, target, visibility)
    
    def __call__(self, output, target, visibility=None):
        return self.compute_loss(output, target, visibility)
    

class Metrics:
    def __init__(self, configs, device='cpu'):
        self.device = device
        self.H = configs.img_size[0]
        self.W = configs.img_size[1]
        self.distance_threshold = configs.ball_size

    def calculate_metrics(self, output, target):
        return heatmap2d_calculate_metrics(output, target, H=self.H, W=self.W)

    def precision_recall_f1(self, output, target):
        return precision_recall_f1_tracknet(output, target, distance_threshold=self.distance_threshold)
    
    def calculate_pck(self, output, target, thresholds=[1,2,3,4,5]):
        return pck_calculation(output, target, thresholds)

    def extract_coordinates(self, heatmap):
        return extract_coords2d(heatmap, H=self.H, W=self.W)
    
    def calculate_rmse(self, x_true, y_true, x_pred, y_pred):
        return calculate_rmse(x_true, y_true, x_pred, y_pred)


class TTLosses:
    def __init__(self, configs, device='cpu'):
        self.loss = Heatmap_Ball_Detection_Loss_Weighted().to(device)

    def compute_loss(self, output, target, visibility=None):
        return self.loss(output, target, visibility)
    
    def __call__(self, output, target, visibility=None):
        return self.compute_loss(output, target, visibility)

class TTMetrics:
    def __init__(self, configs, device='cpu'):
        self.device = device
        self.distance_threshold = configs.ball_size

    def calculate_metrics(self, output, target):
        return heatmap_calculate_metrics(output, target)

    def precision_recall_f1(self, output, target):
        return precision_recall_f1_tracknet(output, target, distance_threshold=self.distance_threshold)

    def extract_coordinates(self, heatmap):
        return extract_coords(heatmap)
    
    def calculate_rmse(self, x_true, y_true, x_pred, y_pred):
        return calculate_rmse(x_true, y_true, x_pred, y_pred)