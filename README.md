Artificial Intelligence A (2024 Spring)
==========================

- 课堂：每周五13:30-16:10，H4101
- 主讲教师：[陈智能](mailto:zhinchen@fudan.edu.cn)
- 助教：[张辉](mailto:23110240067@m.fudan.edu.cn)、[梁月潇](mailto:22210240027@m.fudan.edu.cn)、[叶兴松](mailto:20307130227@fudan.edu.cn)、[王思尹](mailto:20307130223@fudan.edu.cn)
- 讨论交流：微信群、github仓库PR
- 期末考试：2024-06-18 13:00-15:00 [闭卷]

课程实践
----------

- [PJ1](pj1) 基于LR、SVM的蛋白质结构分类 [3.22 - 4.7]
- [PJ2](pj2) 基于CNN的图像分类 [4.12 - 4.26]
- [PJ3](pj3) 人工智能算法挑战赛 [4.26 - 6.2]

# Paper Recommendation

## LM Recommendation
### **1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)**

- **科研团队：** Devlin, Chang, Lee, and Toutanova

- **相关研究：** 掩码语言模型（MLM）、双向变压器、编码器架构

### **2. [Improving Language Understanding by Generative Pre-Training (2018)](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)**

- **科研团队：** Radford and Narasimhan

- **相关研究：** 解码器架构、自回归模型、下一个单词预测、GPT

### **3. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (2019)](https://arxiv.org/abs/1910.13461)**

- **科研团队：** Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, and Zettlemoyer

- **相关研究：** 编码器-解码器架构、去噪预训练、自然语言生成

### **4. [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond (2023)](https://arxiv.org/abs/2304.13712)**

- **科研团队：** Yang, Jin, Tang, Han, Feng, Jiang, Yin, and Hu

- **相关研究：** LLM架构演变、预训练和微调数据、效率提升

### **5. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)](https://arxiv.org/abs/2205.14135)**

- **科研团队：** Dao, Fu, Ermon, Rudra, and Ré

- **相关研究：** 快速注意力机制、内存效率、IO感知

### **6. [Cramming: Training a Language Model on a Single GPU in One Day (2022)](https://arxiv.org/abs/2212.14034)**

- **科研团队：** Geiping and Goldstein

- **相关研究：** 掩码语言模型、单GPU训练、训练效率

### **7. [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)**

- **科研团队：** Hu, Shen, Wallis, Allen-Zhu, Li, L Wang, S Wang, and Chen

- **相关研究：** 参数效率、微调、低秩适应

### **8. [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (2022)](https://arxiv.org/abs/2303.15647)**

- **科研团队：** Lialin, Deshpande, and Rumshisky

- **相关研究：** 参数高效微调、前缀调整、适配器

### **9. [Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)**

- **科研团队：** Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, de Las Casas, Hendricks, Welbl, Clark, Hennigan, Noland, Millican, van den Driessche, Damoc, Guy, Osindero, Simonyan, Elsen, Rae, Vinyals, and Sifre

- **相关研究：** Chinchilla模型、生成任务、线性缩放定律

### **10. [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling (2023)](https://arxiv.org/abs/2304.01373)**

- **科研团队：** Biderman, Schoelkopf, Anthony, Bradley, O'Brien, Hallahan, Khan, Purohit, Prashanth, Raff, Skowron, Sutawika, and van der Wal

- **相关研究：** LLM套件、训练过程分析、架构改进

### **11. [Training Language Models to Follow Instructions with Human Feedback (2022)](https://arxiv.org/abs/2203.02155)**

- **科研团队：** Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, and Lowe

- **相关研究：** 人类反馈、强化学习、InstructGPT

### **12. [Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073)**

- **科研团队：** Yuntao, Saurav, Sandipan, Amanda, Jackson, Jones, Chen, Anna, Mirhoseini, McKinnon, Chen, Olsson, Olah, Hernandez, Drain, Ganguli, Li, Tran-Johnson, Perez, Kerr, Mueller, Ladish, Landau, Ndousse, Lukosuite, Lovitt, Sellitto, Elhage, Schiefer, Mercado, DasSarma, Lasenby, Larson, Ringer, Johnston, Kravec, El Showk, Fort, Lanham, Telleen-Lawton, Conerly, Henighan, Hume, Bowman, Hatfield-Dodds, Mann, Amodei, Joseph, McCandlish, Brown, Kaplan

- **相关研究：** 人工智能对齐、无害系统、自我训练机制

### **13. [Self-Instruct: Aligning Language Model with Self Generated Instruction (2022)](https://arxiv.org/abs/2212.10560)**

- **科研团队：** Wang, Kordi, Mishra, Liu, Smith, Khashabi, and Hajishirzi

- **相关研究：** 自我指导、指令微调、LLM对齐

### **14. [InstructGPT: Aligning Language Models with Human Intent (2022)](https://arxiv.org/abs/2203.02155)**

- **科研团队：** Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, Lowe

- **相关研究：** 指令对齐、人类反馈、InstructGPT

### **15. [LIMA: Less Is More for Alignment (2023)](https://arxiv.org/abs/2305.11206)**

- **科研团队：** Zhou, Qiu, Zhou, Zhang, Hsiao, Chung, Le, and Devlin

- **相关研究：** 简化对齐过程、模型效率

### **16. [RWKV: Reinventing RNNs for the Transformer Era (2023)](https://arxiv.org/abs/2305.13048)**

- **科研团队：** Peng, Yan, and Chen

- **相关研究：** RNN和Transformer结合、性能优化

### **17. [ResiDual: Transformer with Dual Residual Connections (2023)](https://arxiv.org/abs/2305.14464)**

- **科研团队：** Yuan, Chan, and Wang

- **相关研究：** 残差连接、模型架构创新

### **18. [LLaMA: Open and Efficient Foundation Language Models (2023)](https://arxiv.org/abs/2305.12087)**

- **科研团队：** Touvron, Lavril, Izacard, Martinet, Lachaux, Lacroix, Roziere, Goyal, Hambro, Azhar, Rodriguez, Joulin, Grave, and Lample

- **相关研究：** 开源语言模型、高效架构

### **19. [MixReview: Alleviating Memorization in Large Language Models with Active Review (2023)](https://arxiv.org/abs/2305.13872)**

- **科研团队：** Lee, Xu, Deshpande, Taylor, Jia, and Zhang

- **相关研究：** 记忆力减轻、主动复习

### **20. [Q-LoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2305.14314)**

- **科研团队：** Dettmers, Pagnoni, Holtzman, and Zettlemoyer

- **相关研究：** 量化、低秩适应、高效微调


## AI4S Recommendation

### AI+ 生物医药

### **1. [AdaDR 在药物重定位方面的性能优于多个基准方法](https://hyper.ai/news/30434)**

- **科研团队：** 中南大学李敏研究团队

- **相关研究：** Gdataset 数据集、Cdataset 数据集、Ldataset 数据集、LRSSL 数据集、GCNs 框架、AdaDR

- **发布期刊：** Bioinformatics, 2024.01

- **论文链接：** [Drug repositioning with adaptive graph convolutional networks](https://academic.oup.com/bioinformatics/article/40/1/btad748/7467059)

### **2. [基于蛋白质口袋的 3D 分子生成模型——ResGen](https://hyper.ai/news/29026)**

- **科研团队：** 浙大侯廷军研究团队

- **相关研究：** CrossDock2020 数据集、全局自回归、原子自回归、并行多尺度建模、SBMG。比最优技术快 8 倍

- **发布期刊：** Nature Machine Intelligence, 2023.09

- **论文链接：** [ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling](https://www.nature.com/articles/s42256-023-00712-7)

- https://www.science.org/doi/10.1126/science.adg7492)

### **3. [基于图神经网络 (GNN) 开发气味分析 AI](https://hyper.ai/news/25952)**

- **科研团队：** Google Research 的分支 Osmo 公司

- **相关研究：** GS-LF 数据库、GNN、贝叶斯优化算法。在 53% 的化学分子、55% 的气味描述词判断中优于人类

- **发布期刊：** Science, 2023.08

- **论文链接：** [A principal odor map unifies diverse tasks in olfactory perception](https://www.science.org/doi/full/10.1126/science.ade4401)

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10248027)

### AI+ 医疗健康

### **1. [视网膜图像基础模型 RETFound，预测多种系统性疾病](https://hyper.ai/news/28113)**

- **科研团队：** 伦敦大学学院和 Moorfields 眼科医院的在读博士周玉昆等人

- **相关研究：** 自监督学习、MEH-MIDAS 数据集、EyePACS 数据集、SL-ImageNet、SSL-ImageNet、SSL-Retinal。RETFound 模型预测 4 种疾病的性能均超越对比模型

- **发布期刊：** Nature, 2023.08

- **论文链接：** [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x)

### **2. [SVM 优化触觉传感器，盲文识别率达 96.12%](https://hyper.ai/news/26561)**

- **科研团队：** 浙江大学的杨赓和徐凯臣课题组

- **相关研究：** SVM 算法、机器学习、CNN、自适应矩估计算法。优化后的传感器能准确识别 6 种动态触摸模式

- **发布期刊：** Advanced Science, 2023.09

- **论文链接：** [Machine Learning-Enabled Tactile Sensor Design for Dynamic Touch Decoding](https://onlinelibrary.wiley.com/doi/10.1002/advs.202303949)

- https://cardiab.biomedcentral.com/articles/10.1186/s12933-023-01854-z)

### **3. [AI 新脑机技术让失语患者「开口说话」](https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&mid=2247503415&idx=1&sn=f4aafd968bc1a659a92f8ffd076300ed&chksm=fd20387dca57b16be54c285a7c2ffa33a8d71c82a2a84a91b57117f3742dcf7ed8bd4c2324b5&scene=21#wechat_redirect)**

- **科研团队：** 加州大学团队

- **相关研究：** nltk Twitter 语料库、多模态语音神经假体、脑机接口、深度学习模型、Cornell 电影语料库、合成语音算法、机器学习

- **发布期刊：** Nature, 2023.08

- **论文链接：** [A high-performance neuroprosthesis for speech decoding and avatar control](https://www.nature.com/articles/s41586-023-06443-4)

- https://www.nature.com/articles/s41591-023-02640-w)

### AI+ 材料化学

### **1. [高通量计算框架 33 分钟生成 12 万种新型 MOFs 候选材料](https://hyper.ai/news/30269)**

- **科研团队：** 美国阿贡国家实验室 Eliu A. Huerta 研究团队

- **相关研究：** hMOFs 数据集、生成式 AI、GHP-MOFsassemble、MMPA、DiffLinker、CGCNN、GCMC

- **发布期刊：**  Nature, 2024.02

- **论文链接：** [A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture](https://www.nature.com/articles/s42004-023-01090-2)

- https://www.nature.com/articles/s41467-023-40756-2)

### **2. [深度学习工具 GNoME 发现 220 万种新晶体](https://hyper.ai/news/28347)**

- **科研团队：** 谷歌 DeepMind 研究团队

- **相关研究：** GNoME 数据库、GNoME、SOTA GNN 模型、深度学习、Materials Project、OQMD、WBM、ICSD

- **发布期刊：** Nature, 2023.11

- **论文链接：** [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9)

### **3. [深度神经网络+自然语言处理，开发抗蚀合金](https://hyper.ai/news/25891)**

- **科研团队：** 德国马克思普朗克铁研究所的研究人员

- **相关研究：** DNN、NLP。读取有关合金加工和测试方法的文本数据，有预测新元素的能力

- **发布期刊：** Science Advances, 2023.08

- **论文链接：** [Enhancing corrosion-resistant alloy design through natural language processing and deep learning](https://www.science.org/doi/10.1126/sciadv.adg7992)

### AI+ 动植物科学

### **1. [SBeA 基于少样本学习框架进行动物社会行为分析](https://hyper.ai/news/29353)**

- **科研团队：** 中科院深圳先进院蔚鹏飞研究团队

- **相关研究：** PAIR-R24M 数据集、双向迁移学习、非监督式学习、人工神经网络、身份识别模型。在多动物身份识别方面的准确率超过 90%

- **发布期刊：**  Nature Machine Intelligence, 2024.01

- **论文链接：** [Multi-animal 3D social pose estimation, identification and behaviour embedding with a few-shot learning framework](https://www.nature.com/articles/s42256-023-00776-5)

### **2. [利用无人机采集植物表型数据的系统化流程，预测最佳采收日期](https://hyper.ai/news/28303)**

- **科研团队：** 东京大学和千叶大学的研究人员

- **相关研究：** 利润预测模型、分割模型、交互式标注、LabelMe、非线性回归模型、BiSeNet 模型
- **发布期刊：** Plant Phenomics, 2023.09

- **论文链接：** [Drone-Based Harvest Data Prediction Can Reduce On-Farm Food Loss and Improve Farmer Income](https://spj.science.org/doi/10.34133/plantphenomics.0086#body-ref-B4)

### **3. [综述：借助 AI 更高效地开启生物信息学研究](https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&mid=2247504084&idx=1&sn=9e9490226b7d0545a05efba80e60134d&chksm=fd20269eca57af88a2533906881d3606c304002aab9fd0750803bd11439ea46732ea3c91e4ea&scene=21#wechat_redirect)**

- **主要内容：** AI 在同源搜索、多重比对及系统发育构建、基因组序列分析、基因发现等生物学领域中，都有丰富的应用案例。作为一名生物学研究人员，能熟练地将机器学习工具整合到数据分析中，必将加速科学发现、提升科研效率。

### AI+ 农林畜牧业

### **1. [利用卷积神经网络，对水稻产量进行迅速、准确的统计](https://hyper.ai/news/26100)**

- **科研团队：** 京都大学的研究人员

- **相关研究：** 卷积神经网络。CNN 模型可以对不同拍摄角度、时间和时期下得到的农田照片准确分析，得到稳定的产量预测结果

- **发布期刊：** Plant Phenomics, 2023.07

- **论文链接：** [Deep Learning Enables Instant and Versatile Estimation of Rice Yield Using Ground-Based RGB Images](https://spj.science.org/doi/10.34133/plantphenomics.0073)

### **2. [结合实验室观测与机器学习，证明番茄与烟草植物在胁迫环境下发出的超声波能在空气中传播](https://hyper.ai/news/24547)**

- **科研团队：** 以色列特拉维夫大学的研究人员

- **相关研究：** 机器学习模型、SVM、Basic、MFCC、Scattering network、神经网络模型、留一法交叉验证。识别准确率高达 99.7%、4-6 天时番茄尖叫声最大

- **发布期刊：** Cell，2023.03

- **论文链接：** [Sounds emitted by plants under stress are airborne and informative](https://doi.org/10.1016/j.cell.2023.03.009)

### **3. [计算机视觉+深度学习开发奶牛跛行检测系统](https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&mid=2247501193&idx=1&sn=9e8206a22c389b28901b7333fcb239a3&chksm=fd2033c3ca57bad5daea57f8ea4c4c49bb2745d7e661f45e1ff956ed4bae2f5e37d929588adc&scene=21#wechat_redirect)**

- **科研团队：** 纽卡斯尔大学及费拉科学有限公司的研究人员

- **相关研究：** 计算机视觉、深度学习、Mask-RCNN 算法、SORT 算法、CatBoost 算法。准确度可达 94%-100%

- **发布期刊：** Nature, 2023.03

- **论文链接：** [Deep learning pose estimation for multi-cattle lameness detection](https://www.nature.com/articles/s41598-023-31297-1)

### AI+ 气象研究

### **1. [综述：数据驱动的机器学习天气预报模型](https://hyper.ai/news/28124)**

- **主要内容：** 数值天气预报是天气预报的主流方法。它通过数值积分，对地球系统的状态进行逐网格的求解，是一个演绎推理的过程。 2022 年以来，天气预报领域的机器学习模型取得了一系列突破，部分成果可以与欧洲中期天气预报中心的高精度预测匹敌。

### AI+ 天文学

### **1. [PRIMO 算法学习黑洞周围的光线传播规律，重建出更清晰的黑洞图像](https://hyper.ai/news/23698)**

- **科研团队：** 普林斯顿高等研究院研究团队

- **相关研究：** PRIMO 算法、PCA、GRMHD。PRIMO 重建黑洞图像

- **发布期刊：** The Astrophysical Journal Letters, 2023.04

- **论文链接：** [The Image of the M87 Black Hole Reconstructed with PRIMO](https://iopscience.iop.org/article/10.3847/2041-8213/acc32d/pdf)

### **2. [利用模拟数据训练计算机视觉算法，对天文图像进行锐化「还原」](https://mp.weixin.qq.com/s?__biz=MzU3NTQ2NDIyOQ==&mid=2247501028&idx=1&sn=bc1c5ff2d8de935c9b6ba93cb6bc0b7b&chksm=fd2032aeca57bbb8e5884d9572a616c435cc9e2417bf390262635f003fcc67913652a8495731&scene=21#wechat_redirect)**

- **科研团队：** 清华大学及美国西北大学研究团队

- **相关研究：** [Galsim](https://github.com/GalSim-developers/GalSim)、[COSMOS](https://doi.org/10.5281/zenodo.3242143)、计算机视觉算法、CNN、Richardson-Lucy 算法、unrolled-ADMM 神经网络

- **发布期刊：** 皇家天文学会月刊，2023.06

- **论文链接：** [Galaxy image deconvolution for weak gravitational lensing with unrolled plug-and-play ADMM](https://www.nature.com/articles/s41421-023-00543-1)

### AI+ 能源环境

### **1. [强化学习算法提前 300 毫秒预测等离子体撕裂风险](https://hyper.ai/news/30296)**

- **科研团队：** 普林斯顿大学 Egemen Kolemen 研究团队

- **相关研究：** OpenAI Gym 库、DNN、AI controller、EFIT、强化学习

- **发布期刊：**  Nature, 2024.02

- **论文链接：** [Avoiding fusion plasma tearing instability with deep reinforcement learning](https://www.nature.com/articles/s41586-024-07024-9)

### **2. [ECA-Net 模型预测中国未来 70 年的风能利用潜力](https://hyper.ai/news/30119)**

- **科研团队：** 北师大黄国和研究团队

- **相关研究：** ERA5 数据集、月度风速数据、GCM、CNN、ECA-Net

- **发布期刊：** ACS publications, 2024.01

- **论文链接：** [Assessing Climate Change Impacts on Wind Energy Resources over China Based on CMIP6 Multimodel Ensemble](https://pubs.acs.org/doi/abs/10.1021/acs.estlett.3c00829)

### **3. [轻量级模型检测光伏电池缺陷，准确率达 91.74%](https://hyper.ai/news/29898)**

- **科研团队：** 东南大学自动化学院张金霞教授团队

- **相关研究：** NAS、Knowledge Distillation、Normal cells、Reduction cells、DARTS、Teacher-Student 模式

- **发布期刊：** Nature, 2024.03

- **论文链接：** [A lightweight network for photovoltaic cell defect detection in electroluminescence images based on neural architecture search and knowledge distillation](https://arxiv.org/abs/2302.07455)

### AI+ 自然灾害

### **1. [机器学习预测未来 40 年的地面沉降风险](https://hyper.ai/news/30173)**

- **科研团队：** 中南大学柳建新研究团队

- **相关研究：** SAR 数据集、机器学习模型、XGBR、LSTM

- **发布期刊：** Journal of Environmental Management, 2024.02

- **论文链接：** [Machine learning-based techniques for land subsidence simulation in an urban area](https://www.sciencedirect.com/science/article/abs/pii/S0301479724000641?via%3Dihub)

### **2. [语义分割模型 SCDUNet++ 用于滑坡测绘](https://hyper.ai/news/29672)**

- **科研团队：** 成都理工大学刘瑞研究团队

- **相关研究：** Sentinel-2 多光谱数据、NASADEM 数据、滑坡数据、GLFE、CNN、DSSA、DSC、DTL、Transformer、深度迁移学习。交并比提高了 1.91% - 24.42%，F1 提高了 1.26% - 18.54%

- **发布期刊：**  International Journal of Applied Earth Observation and Geoinformation, 2024.01

- **论文链接：** [A deep learning system for predicting time to progression of diabetic retinopathy](https://www.nature.com/articles/s41591-023-02702-z)

### **其他**

### **1. [TacticAI 足球助手战术布局实用性高达 90%](https://hyper.ai/news/30454)**

- **科研团队：** 谷歌 DeepMind 与利物浦足球俱乐部

- **相关研究：** Geometric deep learning、GNN、predictive model、generative model。射球机会提升 13%

- **发布期刊：** Nature, 2024.03

- **论文链接：** [TacticAI: an AI assistant for football tactics](https://www.nature.com/articles/s41467-024-45965-x)

### **2. [去噪扩散模型 SPDiff 实现长程人流移动模拟](https://hyper.ai/news/30069)**

- **科研团队：** 清华大学电子工程系城市科学与计算研究中心、清华大学深圳国际研究生院深圳市泛在数据赋能重点实验室、鹏城实验室的研究人员

- **相关研究：** GC 数据集、UCY 数据集、条件去噪扩散模型、SPDiff、GN、EGCL、LSTM、多帧推演训练算法。5% 训练数据量即可达到最优性能

- **发布期刊：**  Nature, 2024.02

- **论文链接：** [Social Physics Informed Diffusion Model for Crowd Simulation](https://arxiv.org/abs/2402.06680)

### **3. [大语言模型 ChipNeMo 辅助工程师完成芯片设计](https://hyper.ai/news/29134)**

- **科研团队：** 英伟达研究团队

- **相关研究：** 领域自适应技术、NVIDIA NeMo、domain-adapted retrieval models、RAG、supervised fine-tuning with domain-specific instructions、DAPT、SFT、Tevatron、LLM

- **发布期刊：** Journals & Magazines, 2024.03

- **论文链接：** [ChipNeMo: Domain-Adapted LLMs for Chip Design](https://arxiv.org/abs/2311.00176)

### **4. [AlphaGeometry 可解决几何学问题](https://hyper.ai/news/29059)**

- **科研团队：** 谷歌 DeepMind 研究团队

- **相关研究：** neural language model、symbolic deduction engine、语言模型

- **发布期刊：** Nature, 2024.01

- **论文链接：** [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)

### **5. [强化学习用于城市空间规划](https://hyper.ai/news/28917)**

- **科研团队：** 清华大学李勇研究团队

- **相关研究：** 深度强化学习、human–artificial intelligence collaborative 框架、城市规划模型、策略网络、价值网络、GNN。在服务和生态指标上击败了 8 名专业人类规划师

- **发布期刊：** Nature Computational Science, 2023.09

- **论文链接：** [Spatial planning of urban communities via deep reinforcement learning](https://www.nature.com/articles/s43588-023-00503-5)

### **6. [Ithaca 协助金石学家进行文本修复、时间归因和地域归因的工作](https://hyper.ai/news/28140)**

- **科研团队：** DeepMind 和威尼斯福斯卡里大学的研究人员

- **相关研究：** I.PHI 数据集、Ithaca 模型、Kullback-Leibler 散度、交叉熵损失函数。文本修复工作的准确率达到 62%，时间归因误差在 30 年内，地域归因准确率达到 71%

- **发布期刊：** Nature, 2020.03

- **论文链接：** [Restoring and attributing ancient texts using deep neural networks](https://www.nature.com/articles/s41586-022-04448-z)

For more, reach [hyperai/awesome-ai4s](https://github.com/hyperai/awesome-ai4s).
