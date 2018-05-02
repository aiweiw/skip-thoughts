#!/usr/bin/evn python
# coding:utf-8
import sys
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def load_labels(anyou,label_map):
    #anyou = 10 第一级分类
    #设置分类标签
    label_count = 0
    #设置训练和测试目录
    if  'childlist' in anyou.keys():
        for child in anyou.get("childlist"):
            #设置分类标签
            label_count += 1
            result = "__label__" + str(label_count) + " , "
            label_map[anyou.get("DM") + "__label__" + str(label_count)] = child
            #配置训练和测试语料
            grandson_list = []
            #print(child)
            #递归获取所有后代节点列表
            get_all_grandsons_id(child,grandson_list)
            #递归装载子节点的分类
            load_labels(child,label_map)
        anyou['label_count'] = label_count

        #print(anyou)


def get_all_grandsons_id(child,grandson_list):
     grandson_list.append(child.get("DM"))
     #如果还有子节点，继续往下查找
     if 'childlist' in child.keys():
         for grand_son in child.get("childlist"):
             get_all_grandsons_id(grand_son,grandson_list)


def getXmlChild(xmlnode,cur_node,nodemap,firstlist):
    for child in xmlnode:
        # print(child.tag, "---", child.attrib)
        fatherid = child.attrib.get("FM")
        selfid = child.attrib.get("DM")
        if selfid  in nodemap.keys():
            if int(selfid) > 0:
                #print "duplicate id " + selfid
                continue
        # 生成新节点
        newnode = {}
        newnode["tag"] = child.tag
        nodemap[selfid] = newnode

        for (k, v) in child.attrib.items():
            newnode[k] = v
        if (fatherid == '0' or fatherid == '-1' or fatherid == None):  # 一级子节点
            firstlist.append(newnode)
        else:
            fathernode = nodemap[fatherid]
            # print(newnode)
            if 'childlist' in fathernode.keys():
                fathernode['childlist'].append(newnode)
            else:
                fathernode['childlist'] = []
                fathernode['childlist'].append(newnode)


def loadConfig(file,firstlist,nodemap,label_map):
    try:
        tree = ET.parse(file)  # 打开xml文档
        # root = ET.fromstring(country_string) #从字符串传递xml
        root = tree.getroot()  # 获得root节点
    except Exception as e:
        print("Error:cannot parse file:",file)
        sys.exit(1)
    jsonroot = {}
    print(root.tag, "---", root.attrib)
    jsonroot["tag"] = root.tag
    #firstlist = []
    #nodemap = {}
    #label_map = {}

    for child in root:
        # print(child.tag, "---", child.attrib)
        fatherid = child.attrib.get("FM")
        selfid = child.attrib.get("DM")
        if selfid in nodemap.keys() :
            print "duplicate ndoe_id " + str(selfid)
            #continue
        # 生成新节点
        newnode = {}
        newnode["tag"] = child.tag
        nodemap[selfid] = newnode
        getXmlChild(child, newnode, nodemap, firstlist)

        for (k, v) in child.attrib.items():
            newnode[k] = v
        if (fatherid == '0' or fatherid == '-1'):  # 一级子节点
            firstlist.append(newnode)
        else:
            fathernode = nodemap[fatherid]
            # print(newnode)
            if 'childlist' in fathernode.keys():
                fathernode['childlist'].append(newnode)
            else:
                fathernode['childlist'] = []
                fathernode['childlist'].append(newnode)

    for anyou in firstlist:
        load_labels(anyou,label_map)

