import json
import os
import os.path as osp
from mmengine.dist import master_only
from pycocotools.coco import COCO
from ._ucit_eval_coco import COCOEvalCap

from .base_eval_dataset import BaseEvalDataset


class UcitBaseEvalDataset(BaseEvalDataset):
    def create_output_file(self, results, output_file):
        outputs = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred_dict in results:
                index = pred_dict['index']
                gt_data  = self.data[index]
                to_write_data = pred_dict
                to_write_data.update(
                        {
                            "question_id": gt_data['question_id'],
                            "text": gt_data['text'],
                            "answer": gt_data['answer'],
                            "prediction": pred_dict['prediction'],
                            "metadata": {},
                        }
                )
                f.write(json.dumps(to_write_data)+ "\n")
                outputs.append({
                    "gt": gt_data['answer'],
                    "pred":  pred_dict['prediction']
                })
        return outputs


    @master_only
    def evaluate(self, results, work_dir):
        results.sort(key=lambda e: e['index'])

        if not osp.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)

        assert len(results) == len(self.data)
        output_file = osp.join(work_dir, "output.jsonl")
        outputs = self.create_output_file(results, output_file)

        total, correct = 0, 0
        for out in outputs:
            correct += out['gt'].upper() == out['pred'].upper()
            total += 1
        accuracy = correct / total * 100
        
        print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, accuracy))
        metric_file = osp.join(work_dir, "metric.txt")
        with open(metric_file, 'w') as f:
            f.write('Dataset: {}\nSamples: {}\nAccuracy: {:.2f}%\n'.format(self.meta_info.get('name'), total, accuracy))
        return {"acc": accuracy}



def create_coco_type(results, output_path):
    total = len(results)
    coco_results = []
    image_id = 1
    for result in results:
        pred = result['prediction']
        coco_results.append({
            "image_id": int(image_id),  # 确保 image_id 是整数类型
            "caption": pred
        })
        image_id += 1
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total


class UcitCaptionEvalDataset(UcitBaseEvalDataset):
    @master_only
    def evaluate(self, results, work_dir):
        if not osp.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            
        results.sort(key=lambda e: e['index'])

        assert len(results) == len(self.data)
        output_file = osp.join(work_dir, "output.jsonl")
        outputs = self.create_output_file(results, output_file)

        coco_res_file = osp.join(work_dir, "pred_coco_type.json")
        create_coco_type(results, coco_res_file)

        coco_anno_file = self.meta_info.get("coco_anno_file", None)
        coco = COCO(coco_anno_file)
        coco_res = coco.loadRes(coco_res_file)

        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.evaluate()

        metrics_to_print = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
        metrics = []
        for metric, score in coco_eval.eval.items():
            if metric in metrics_to_print:
                score_percentage = score * 100.
                print(f"{metric}: {score_percentage:.2f}")
                metrics.append(score_percentage)
        avg_metric = sum(metrics) / len(metrics)
        print('Samples: {}\nAverage: {:.2f}%\n'.format(len(results), avg_metric))

        metric_file = osp.join(work_dir, "metric.txt")
        with open(metric_file, 'w') as f:
            f.write('Dataset: {}\nSamples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                self.meta_info.get('name'), len(results), metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], sum(metrics) / len(metrics)))
    
        return {"acc": avg_metric}

