# pan_aca
Improvements to PAN   
   
Multiple dataset processing has been added to the code, including LSVT, ArT, COCO, mlt.   
   
The method uses an adaptive channel attention mechanism (ACA) to obtain more representative textual features through local cross-channel interactions, improving the performance of deep convolutional neural networks. The feature enhancement pyramid module (FPEM) is used to fuse low-level and high-level information to further enhance features at different scales. Meanwhile, to address the scale variation problem of long texts, a weighted aware loss (WAL) is proposed to enhance robustness by adjusting the weights of text instances of different sizes. Finally, the experiments verify the superiority of the method on CTW1500 and MSRA-TD500 standard datasets.   
