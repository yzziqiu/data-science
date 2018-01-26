# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request

class ClasscentralSpider(scrapy.Spider):
    name = 'classcentral'
    allowed_domains = ['coding.imooc.com']
    start_urls = ['http://coding.imooc.com/']

    def parse(self, response):
        # 解析实战课程列表页
        for sel in response.css('div.shizhan-course-wrap > a'):
            course_id = sel.xpath('./@href').re_first('(\d+).html')
            url = response.urljoin(sel.xpath('./@href').extract_first())
            name = sel.css('p.shizan-name::attr(title)').extract_first()

            base_item = {
                'id': course_id,
                '课程名': name,
            }

            url = 'class/evaluation'.join(url.split('class'))
            yield Request(url, callback=self.parse_comment, meta={'base_item': base_item})

        next_url = response.xpath('//div[@class="page"]/a[contains(., "下一页")]/@href').extract_first()

        if next_url:
            yield Request(response.urljoin(next_url), callback=self.parse)

    def parse_comment(self, response):
        # 解析实战评价页面
        base_item = response.meta['base_item']

        if not response.meta.get('recommend', False):
            response.meta['recommend'] = True
            for sel in response.css('div.evaluation-recommend li p.content::attr(title)'):
                item = base_item.copy()
                item['评论'] = sel.extract()
                yield item

        for sel in response.css('ul.cmt-list p.cmt-txt::text'):
            item = base_item.copy()
            item['评论'] = sel.extract()
            yield item

        next_url = response.xpath('//div[@class="page"]/a[contains(., "下一页")]/@href').extract_first()
        if next_url:
            yield Request(response.urljoin(next_url), callback=self.parse_comment, meta=response.meta)
