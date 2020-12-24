#接口文档
###一、简介
###二、接口说明
####消息队列
|接口名称/TOPIC|接口格式|接口介绍|
|---|---|---|
|warning|{‘type’:1, 'str':'wew'}|用于统一对项目中的报警进行管理|
|numQueue|{‘numberOfQueue’:12}|统计当前选定区域排队人数|
|mostStaningTime|{‘mostStaningTime’:123}|统计当前区域最久停留时间|
|crossRegion|{'numArea':3,'flow':[12,23,0,44,0,12,12,34,0]}|统计人员流动|
|genderRate|{'男性':12, '女性': 13}|统计男女比例|
|ageRate|{'age':[12,15,32,14]}|统计年龄比例|
|latestDay|{'population':[12,13,543,12]}|统计最近多个小时内人数8：00～17：00|
|faceAttr|{'img':'base64image','age':13,'gender':13}|主屏幕上滚动显示(检测新的人脸)|
|faceAttr|{'infos':[{'img':'base64image','age':13,'gender':13}]}|主屏幕上滚动显示（按周期显示人脸）|
|abnormal|{‘name’：‘knife’,'img':'base64image'}|每当检测到异常物品，裁剪图片并发送类别|
|leftover|{‘name':'bag','img':'base64image'}|遗留物品检测|
|managerStatus|{'status':'在岗/暂离/离岗'}|大堂经理状态|
|numRegion|{'infos':[{'name':'region1','numPerson':12,'avgStayTime':123},]}|区域对应的人数|
以上均采用websocket与消息队列实现前后端信息推送,各模型负责部分应将统计数据发至消息队列指定topic
####后端接口
|接口名称|请求内容|接口介绍|请求类型|
|---|---|---|---|
|setQueueSize| |设置最大等待人数|get|
|setEntrySize| |设置最大进入人数|get|
|setBankCapacity| |设置最大停留人数|get|
|setWaitTime| |设置最长停留时间|get|
|setWaitNumber| |设置最大等待人数|get|
|setLeaveTime| |设置最长离岗时间|get|
|setContactTime| |设置最长两人接触时间|get|
|getLastWeekNum| |获取最近七天人数|post|
|backgroundShot| |遗留物品检测背景拍摄|get|

##三、对接接口格式
###四、报警类别及对应标识

