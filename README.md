## requirements
- python==3.6
- tensorflow-gpu==1.14.0
- cuda10.0
- h5py==2.10.0
- Keras==2.2.5
- opencv-python
- pytorch==1.2.0
- torchvision==0.4.0

## 12.31
- add SOT module: it can track a detection by click one person

## 1.4
- [ ] update feature list by track

- [ ] add kalmen filter to avoid far by similar detection
    - cache the last track's bounding box, then check nms to avoid long-term disappear
- [ ] update associate strategy
    - associate for more than 3 times can be regard as associate successfully(**the judgement criteria**)
- not track
    - 使用MGN提取选择目标的特征
    - 匹配策略
        - IOU， 当记录的bbox存在时，如果计算IOU为0,代表错误匹配
        - 相似度未达到阈值，则表示错误匹配
    - 根据一些限制来更新特征或平滑特征 
        - 如果匹配目标OK并且目标附近无其他目标引起遮挡，表示检测效果应该良好，则更新特征
    - 离岗判断条件
        - 如果连续帧均匹配上，那么表示出现了目标，否则如果在一定时间段内未出现目标，则表示存在离岗情况
- track
    - MGN提取特征
    - 根据卡尔慢滤波进行预测，来跟踪目标
    - 可能已为IDS问题导致追踪错误

## 1.5
跨域问题
django安装第三方插件后，前端受到了响应的文件但不能显示
curl可以获取网页

用nginx进行转发，用8081端口转发到127.0.0.1：8000仍然有跨域问题
cors missing allow origin

通过在ngnix配置相关add_header添加（Access-Control-Allow-Origin、
Access-Control-Allow-Headers、Access-Control-Allow-Methods）

## 1.6 
完成通过前端发送的post请求，实现模型中参数的更新，主要实现通过点击改变追踪目标的功能