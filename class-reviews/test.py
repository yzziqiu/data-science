from aip import AipNlp
import csv
import pandas as pd
from pandas.core.frame import DataFrame

""" 你的 APPID AK SK """
# 利用百度云提供的API接口实现情感分析
APP_ID = '10752821'
API_KEY = 'H2unP0GujMgWGLuZt7dYjZ2R'
SECRET_KEY = 'd3GoydEcLDjQYWPecpZSvtrI7G8cpBWG'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# 写入文件
def output():
    urls = []
    with open('class.csv', "r") as f:
        reader = csv.reader(f)
        for row in reader:
            urls.append(row[2])
    return urls

# 对读入的数据进行情感分析，将其得到的结果解析成标准JSON格式数据，并保存在一个新的dict中
def sentimentClassify():
    x = output()
    i=1
    pp=[]
    np=[]
    con=[]
    sen=[]
    all={}
    for i in range(10640):
        text=x[i]
        # 通过百度提供的接口方法进行情感倾向提取
        result=client.sentimentClassify(text);
        print(result)
        # 如果解析错误则填写上空值,使得程序不会出错而停止运行
        if "error_code" in result.keys():
            pp.append(" ")
            np.append(" ")
            con.append(" ")
            sen.append(" ")
            all['positive_prob'] = pp
            all['negative_prob'] = np
            all['confidence'] = con
            all['sentiment'] = sen
        else:
            data = result['items']
            items = data[0]
            positive_prob = items['positive_prob']
            pp.append(positive_prob)
            negative_prob = items['negative_prob']
            np.append(negative_prob)
            confidence = items['confidence']
            con.append(confidence)
            sentiment = items['sentiment']
            sen.append(sentiment)
            all['positive_prob'] = pp
            all['negative_prob'] = np
            all['confidence'] = con
            all['sentiment'] = sen
    return all

# 将得到的dict存储到原始的CSV文件中，方便后续进行分析
def add(ulist):
    csv_input = pd.read_csv('class.csv',encoding='utf-8')
    pp = DataFrame(ulist['positive_prob'])
    csv_input["positive_prob"] = pp
    csv_input.to_csv('class.csv', index=False,encoding='utf-8')
    np = DataFrame(ulist['negative_prob'])
    csv_input["negative_prob"] = np
    csv_input.to_csv('class.csv', index=False, encoding='utf-8')
    con = DataFrame(ulist['confidence'])
    csv_input["confidence"] = con
    csv_input.to_csv('class.csv', index=False, encoding='utf-8')
    sen = DataFrame(ulist['sentiment'])
    csv_input["sentiment"] = sen
    csv_input.to_csv('class.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    ALL=sentimentClassify()
    add(ALL)
