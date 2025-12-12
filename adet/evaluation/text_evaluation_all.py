import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from adet.evaluation import text_eval_script
from adet.evaluation import text_eval_script_ic15
import zipfile
import pickle
import editdistance
import cv2
from adet.modeling.model.language import get_language_tokenizer_BART

class TextEvaluator():
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )

        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dict = cfg.MODEL.TRANSFORMER.CUSTOM_DICT

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self.bert_tokenizer = get_language_tokenizer_BART()
        self.dataset_name = dataset_name
        self.submit = False
        # use dataset_name to decide eval_gt_path
        self.lexicon_type = cfg.TEST.LEXICON_TYPE
        # use dataset_name to decide eval_gt_path
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/gt_totaltext.zip"
            self._word_spotting = True
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/gt_ctw1500.zip"
            self._word_spotting = False
        elif "ic15" in dataset_name:
            self._text_eval_gt_path = "datasets/gt_icdar2015.zip"
            self._word_spotting = False
        else:
            self._text_eval_gt_path = ""

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results"):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            a = [c for c in s if ord(c) < 128]
            outa = ''
            for i in a:
                outa +=i
            return outa

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        for i in range(len(data[ix]['polys'])):
                            if "ctw1500" in self.dataset_name:
                                # there are many boundary points on each side, 'float' type is used for ctw1500
                                # the original implementation in Adelaidet adopts 'int'
                                outstr = outstr + str(float(data[ix]['polys'][i][0])) +\
                                         ','+str(float(data[ix]['polys'][i][1])) +','
                            else:
                                outstr = outstr + str(int(data[ix]['polys'][i][0])) + \
                                         ',' + str(int(data[ix]['polys'][i][1])) + ','

                        ass = str(data[ix]['rec'])
                        if len(ass)>=0: # 
                            outstr = outstr + str(round(data[ix]['score'], 3)) +',####'+ass+'\n'	
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        if not os.path.isdir(dirn):
            os.mkdir(dirn)

        buckets = {}  
        for line in fres:
            line = line.strip()
            s = line.split(': ')
            filename = '{:07d}.txt'.format(int(s[0]))
            ptr = s[1].split(',####')
            score = float(ptr[0].split(',')[-1])
            cors = ','.join(ptr[0].split(',')[:-1])
            rec  = ptr[1]
            buckets.setdefault(filename, []).append((score, cors, rec))

        for filename, items in buckets.items():
            def key_fn(t):
                sc, cors, _ = t
                xs, ys = cors.split(',')[0], cors.split(',')[1]
                return (-sc, float(xs), float(ys))
            items.sort(key=key_fn)  
            out = os.path.join(temp_dir, filename)
            with open(out, 'w') as fout:
                for score, cors, rec in items:
                    fout.write(f"{cors},####{rec}\n")

        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_"+temp_dir
        output_file_full = "full_final_"+temp_dir
        if not os.path.isdir(output_file_full):
            os.mkdir(output_file_full)
        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        files = glob.glob(origin_file+'*.txt')
        files.sort()
        if "totaltext" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = 'datasets/totaltext/tt_lexicon.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/totaltext/tt_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
        elif "ctw1500" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = 'datasets/ctw1500/weak_voc_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/ctw1500/weak_voc_pair_list.txt', 'r')
                pairs = dict()
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
                    pairs[line.upper()] = line
        elif "ic15" in self.dataset_name:
            if self.lexicon_type==1: 
                # generic lexicon
                lexicon_path = 'datasets/ic15/GenericVocabulary_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/ic15/GenericVocabulary_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
            if self.lexicon_type==2:
                # weak lexicon
                lexicon_path = 'datasets/ic15/ch4_test_vocabulary_new.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/ic15/ch4_test_vocabulary_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)
        elif "inversetext" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = 'datasets/inversetext/inversetext_lexicon.txt'
                lexicon_fid=open(lexicon_path, 'r')
                pair_list = open('datasets/inversetext/inversetext_pair_list.txt', 'r')
                pairs = dict()
                for line in pair_list.readlines():
                    line=line.strip()
                    word = line.split(' ')[0].upper()
                    word_gt = line[len(word)+1:]
                    pairs[word] = word_gt
                lexicon_fid=open(lexicon_path, 'r')
                lexicon=[]
                for line in lexicon_fid.readlines():
                    line=line.strip()
                    lexicon.append(line)

        def find_match_word(rec_str, pairs, lexicon=None):
            rec_str = rec_str.upper()
            dist_min = 100
            dist_min_pre = 100
            match_word = ''
            match_dist = 100
            for word in lexicon:
                word = word.upper()
                ed = editdistance.eval(rec_str, word)
                length_dist = abs(len(word) - len(rec_str))
                dist = ed
                if dist<dist_min:
                    dist_min = dist
                    match_word = pairs[word]
                    match_dist = dist
            return match_word, match_dist
        def _signed_area(poly):
            s = 0.0
            n = len(poly)
            for k in range(n):
                x1, y1 = poly[k]
                x2, y2 = poly[(k+1) % n]
                s += x1 * y2 - y1 * x2
            return 0.5 * s
        def _dedup_consecutive(points):
            out, prev = [], None
            for p in points:
                if prev is None or (p[0] != prev[0] or p[1] != prev[1]):
                    out.append(p)
                    prev = p
            return out

        def _ensure_clockwise(points):
            return list(points)[::-1] if _signed_area(points) >= 0 else points             

        for i in files:
            if "ic15" in self.dataset_name:
                out = output_file + 'res_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                out_full = output_file_full + 'res_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                if self.lexicon_type==3:
                    lexicon_path = 'datasets/ic15/new_strong_lexicon/new_voc_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt'
                    lexicon_fid=open(lexicon_path, 'r')
                    pair_list = open('datasets/ic15/new_strong_lexicon/pair_voc_img_' + str(int(i.split('/')[-1].split('.')[0])) + '.txt')
                    pairs = dict()
                    for line in pair_list.readlines():
                        line=line.strip()
                        word = line.split(' ')[0].upper()
                        word_gt = line[len(word)+1:]
                        pairs[word] = word_gt
                    lexicon_fid=open(lexicon_path, 'r')
                    lexicon=[]
                    for line in lexicon_fid.readlines():
                        line=line.strip()
                        lexicon.append(line)
            else:
                out = i.replace(origin_file, output_file)
                out_full = i.replace(origin_file, output_file_full)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            fout_full = open(out_full, 'w')
            from shapely.ops import unary_union
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',####')
                rec  = ptr[1]
                cors = ptr[0].split(',')
                assert (len(cors) % 2 == 0), 'cors invalid.'
                pts = [(float(cors[j]), float(cors[j+1])) for j in range(0, len(cors), 2)]

                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e); print('Invalid detection in {} line {} -> drop'.format(i, iline))
                    continue

                if not pgt.is_valid:
                    pgt = pgt.buffer(0)
                    if not pgt.is_valid:
                        arr = np.asarray(pts, dtype=np.float32)
                        try:
                            hull = cv2.convexHull(arr).squeeze(1)
                            if hull.ndim == 1 or hull.shape[0] < 4:
                                print('Invalid detection in {} line {} -> drop'.format(i, iline))
                                continue
                            rect = cv2.minAreaRect(hull)
                            box  = cv2.boxPoints(rect)
                            pts  = [(float(x), float(y)) for x, y in box]
                            pgt  = Polygon(pts)
                        except Exception as e:
                            print(e); print('Invalid detection in {} line {} -> drop'.format(i, iline))
                            continue

                if "totaltext" in self.dataset_name or "ctw1500" in self.dataset_name:
                    pts = [(float(x), float(y)) for x, y in pts]
                    pts = _dedup_consecutive(pts)
                    if len(pts) < 4:
                        continue

                    pts = _ensure_clockwise(pts)
                    pts = [(int(round(x)), int(round(y))) for x, y in pts]
                    pts = _dedup_consecutive(pts)
                    if len(pts) < 4:
                        continue

                    is_simple = False
                    try:
                        is_simple = LinearRing(pts).is_simple
                    except Exception:
                        is_simple = False

                    if not is_simple:
                        poly = Polygon(pts)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if isinstance(poly, Polygon) and (not poly.is_empty):
                            ext = list(poly.exterior.coords)[:-1]
                            if len(ext) >= 4:
                                pts = [(int(round(x)), int(round(y))) for x, y in ext]
                                pts = _dedup_consecutive(pts)
                                if len(pts) >= 4:
                                    pts = _ensure_clockwise(pts)
                                    try:
                                        if not LinearRing(pts).is_simple:
                                            continue
                                    except Exception:
                                        continue
                        else:
                            continue
                else:
                    arr = np.asarray(pts, dtype=np.float64)
                    area2 = np.sum(arr[:,0]*np.roll(arr[:,1], -1) - arr[:,1]*np.roll(arr[:,0], -1))
                    if area2 < 0:
                        pts.reverse()
                    if "ctw1500" in self.dataset_name:
                        pts = [(float(x), float(y)) for x, y in pts]
                    else:
                        pts = [(int(round(x)), int(round(y))) for x, y in pts]

                outstr = ''
                for x, y in pts:
                    if "ctw1500" in self.dataset_name:
                        outstr += f"{float(x)},{float(y)},"
                    else:
                        outstr += f"{int(x)},{int(y)},"
                outstr = outstr[:-1]
                if "ic15" in self.dataset_name:
                    rec = rec.replace(',', '')
                    outstr = outstr + ',' + rec
                else:
                    outstr = outstr + ',####' + rec
                fout.writelines(outstr + '\n')   
                if self.lexicon_type is None:
                    rec_full = rec
                else:
                    match_word, match_dist = find_match_word(rec, pairs, lexicon)
                    # if match_dist < 1.5:
                    if "ctw1500" in self.dataset_name:
                        # edit_th = 0.3
                        # match_dist = float(match_dist/len(match_word))
                        # print(match_dist,edit_th,'match_dist,edit_th')
                        edit_th = 3.5 
                        # print(match_dist,edit_th,'match_dist,edit_th')
                    else:    
                        edit_th = 1.5
                    
                    # if match_dist < 3.5 :
                    if match_dist < 10000:
                        rec_full = match_word
                        if "ctw1500" in self.dataset_name:
                            coord_str = ",".join(f"{float(x)},{float(y)}" for x, y in pts)
                        else:
                            coord_str = ",".join(f"{int(x)},{int(y)}" for x, y in pts)
                        if "ic15" in self.dataset_name:
                            safe_rec_full = rec_full.replace(",", "")
                            out_full_line = f"{coord_str},{safe_rec_full}\n"
                        else:
                            out_full_line = f"{coord_str},####{rec_full}\n"

                        fout_full.write(out_full_line)
            fout.close()
            fout_full.close()

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        if "ic15" in self.dataset_name:
            os.system('zip -r -q -j '+'det.zip'+' '+output_file+'/*')
            os.system('zip -r -q -j '+'det_full.zip'+' '+output_file_full+'/*')
            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"
        else:
            os.chdir(output_file)
            zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('./', zipf)
            zipf.close()
            os.chdir("../")

            os.chdir(output_file_full)
            zipf_full = zipfile.ZipFile('../det_full.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('./', zipf_full)
            zipf_full.close()
            os.chdir("../")
            # clean temp files

            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"
    
    def evaluate_with_official_code(self, result_path, gt_path):
        if "ic15" in self.dataset_name:
            return text_eval_script_ic15.text_eval_main_ic15(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)
        else:
            return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        PathManager.mkdirs(self._output_dir)
        if self.submit:
            file_path = os.path.join(self._output_dir, self.dataset_name+"_submit.txt")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                for prediction in predictions:
                    write_id = "{:06d}".format(prediction["image_id"]+1)
                    write_img_name = "test_"+write_id+'.jpg\n'
                    f.write(write_img_name)
                    if len(prediction["instances"]) > 0:
                        for inst in prediction["instances"]:
                            write_poly, write_text = inst["polys"], inst["rec"]
                            if write_text == '':
                                continue
                            if "totaltext" in self.dataset_name:
                                # enforce clockwise for Total-Text
                                if LinearRing(write_poly).is_ccw:
                                    write_poly.reverse()
                            else:
                                if not LinearRing(write_poly).is_ccw:
                                    write_poly.reverse()
                            # if not LinearRing(write_poly).is_ccw:
                            #     write_poly.reverse()
                            write_poly = np.array(write_poly).reshape(-1).tolist()
                            write_poly = ','.join(list(map(str,write_poly)))
                            f.write(write_poly+','+write_text+'\n')
                f.flush()
            self._logger.info("Ready to submit results from {}".format(file_path))
        else:
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
            file_path = os.path.join(self._output_dir, "text_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        self._results = OrderedDict()
        # eval text
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)

        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir)
        result_path, result_path_full = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path) # None 
        text_result["e2e_method"] = "None-" + text_result["e2e_method"]
        dict_lexicon = {"1": "Generic", "2": "Weak", "3": "Strong"}
        text_result_full = self.evaluate_with_official_code(result_path_full, self._text_eval_gt_path) # with lexicon
        text_result_full["e2e_method"] = dict_lexicon[str(self.lexicon_type)] + "-" + text_result_full["e2e_method"]
        os.remove(result_path)
        os.remove(result_path_full)
        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        result = text_result["det_only_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        result = text_result["e2e_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        result = text_result_full["e2e_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}

        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs['width']
        height = inputs['height']
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        pnts = instances.bd.numpy()
        recs = instances.recs.numpy()

        results = []
        for pnt, rec, score in zip(pnts, recs, scores):
            poly = self.pnt_to_polygon(pnt)
            if 'ic15' in self.dataset_name or 'rects' in self.dataset_name:
                poly = polygon2rbox(poly, height, width)
            s = self.ctc_decode(rec)
            result = {
                "image_id": img_id,
                "category_id": 1,
                "polys": poly,
                "rec": s,
                "score": score,
            }
            results.append(result)
        return results

    def pnt_to_polygon(self, ctrl_pnt):
        ctrl_pnt = np.hsplit(ctrl_pnt, 2)
        ctrl_pnt = np.vstack([ctrl_pnt[0], ctrl_pnt[1][::-1]])
        return ctrl_pnt.tolist()

    def ctc_decode(self, rec):
        s = ''
        # print(rec,'rec')
        out_text = self.bert_tokenizer.decode(rec)
        #need to do some replacement       
        print(out_text)
        out_text =  out_text.replace("[CLS]","")
        out_text =  out_text.replace("[UNK]","")
        # out_text =  out_text.replace(" ","")
        out_text =  out_text.replace("[SEP]","")
        out_text =  out_text.replace("[PAD]","")
        out_text =  out_text.replace("<s>","")
        out_text =  out_text.replace("<mask>","")
        out_text =  out_text.replace("<unk>","")
        out_text =  out_text.replace("</s>","")
        out_text =  out_text.replace("<pad>","")
        print(out_text)
        if len(out_text) == 0:
            return '###'
        return out_text
            
def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1,2)
    pts = pts.tolist()
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]
