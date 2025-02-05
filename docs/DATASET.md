# Data instruction

-----

To facilitate reproduction, we provide access to the processed dataset. You can acquire it only upon request. Before doing so, you must first obtain permission from the original provider. We list the procedure for obtaining each dataset here:

* __THuman2.0__: Fill out the [request form](https://github.com/ytrock/THuman2.0-Dataset/blob/main/THUman2.1_Agreement.pdf) and send it to Yebin Liu (liuyebin@mail.tsinghua.edu.cn) and cc Tao Yu (ytrock@126.com) to request the download link.  By requesting the link, you acknowledge that you have read the agreement, understand it, and agree to be bound by it. If you do not agree with these terms and conditions, you must not download and use the Dataset.
* __XHuman__: Follow [the instruction](https://xhumans.ait.ethz.ch/) to get the access to the XHuman dataset.
* __AIST++__: The original dataset can be found [here](https://google.github.io/aistplusplus_dataset/download.html).

After getting the access, please email Jing Wen (jw116@illinois.edu) with the title "Request for processed data". Please attach the screenshot of the comfirmation email from authors of the original datasets.

Unzip the datasets; The directory should follow the following structure:
```
├── $ROOT/data
    ├── thuman2.0
        ├── view3_train
        ├── view3_val
        ├── view5_train
        ├── view5_val
        ├── train.json
        ├── val.json
        ├── view3_val_meta.json
        ├── view5_val_meta.json
    ├── xhuman
        ├── view3_val
        ├── val.json
    ├── aistpp
        ├── view5_trainval
        ├── train.json
```
