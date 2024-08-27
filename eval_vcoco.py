import sys
sys.path.append('/zengyq/dataset/v-coco')
from vsrl_eval import VCOCOeval
import utils

if __name__ == "__main__":
    vsrl_annot_file = "/zengyq/dataset/v-coco/data/vcoco/vcoco_test.json"
    coco_file = "/zengyq/dataset/v-coco/data/instances_vcoco_all_2014.json"
    split_file = "/zengyq/dataset/v-coco/data/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "/zengyq/HOI/ADA-CM-24-Latest/checkpoints/vcoco/cache.pkl"

    print(f"Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)