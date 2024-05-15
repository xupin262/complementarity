# Leveraging Sample Complementarity: A Novel Ensemble Strategy

DFUC2022.

IJCNN2024.

本文代码基于mmsegmentation（https://github.com/open-mmlab/mmsegmentation）实现。

1.根据官方文档使用mmsegmentation并配置环境conda环境。

2.修改DFUC2022数据集的mask标签，使其值仅为0(背景)或1(溃疡).

3.将本项目给定的DFUC2022数据集配置文件dfuc.py放在mmseg/datasets/文件中，并在mmseg/datasets/__init__.py中添加相关信息：

from .dfuc import DFUCDataset

__all__ = [……此处添加,'DFUCDataset']

4.训练各个模型

5.测试模型效果。

若要该模型参与集成，则需要在集成前保存各个候选模型的测试数据，方法如下：
在miniconda3/envs/该项目的conda环境名/lib/python3.8/site-packages/mmengine/runner/loops.py文件中修改run()函数信息（主要是加上保存模型预测的图片结果权重到文件out_save_path中的代码）

    # 修改这里
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        # 保存模型预测的图片结果权重到文件out_save_path中
        out_save_path = self._runner._test_evaluator["output_dir"]+'_predict.pt' # 'work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_dfuc-512x512_nq100/mask2former_swin-l_nq100_112k_predict.pt'  #####################################################################
        tensor_dict = {}   
        for idx, data_batch in enumerate(self.dataloader):
            outputs = self.run_iter(idx, data_batch)
            # import pdb; pdb.set_trace()
            img_name = outputs[0].img_path.split('/')[-1].split('.')[0]
            tensor_dict[img_name] = outputs[0].seg_logits.data
        # 保存模型参数到文件
        # torch.save(tensor_dict, out_save_path)   #####################################################################
        
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

6.模型集成
使用tools文件夹中的model_ensemble_wAverage.py进行加权平均集成，使用model_ensemble_HB.py进行互补集成。




