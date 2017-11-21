# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class BaidustocksPipeline(object):
    def process_item(self, item, spider):
        return item

#每个pipelines类中有三个方法
class BaidustocksInfoPipeline(object):
    #当一个爬虫被调用时，对应的pipelines启动的方法
    def open_spider(self, spider):
        self.f = open('BaiduStockInfo.txt', 'w')
    #一个爬虫关闭或结束时的pipelines对应的方法
    def close_spider(self, spider):
        self.f.close()
    #对每一个Item项进行处理时所对应的方法，也是pipelines中最主体的函数
    def process_item(self, item, spider):
        try:
            line = str(dict(item)) + '\n'
            self.f.write(line)
        except:
            pass
        return item
