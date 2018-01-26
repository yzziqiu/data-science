# data-explore.py
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud

import jieba.analyse
df = pd.read_json('classcentral.json')

group = df.groupby('id')
f = lambda s: dict(jieba.analyse.extract_tags(s.str.cat(), 10, withWeight=True, allowPOS=('a', 'an')))
res = group['评论'].apply(f)

res_group = res.groupby('id')
n = len(res_group)
figure, axes = plt.subplots(n, 2, figsize = (6*2, 4*n))
figure.subplots_adjust(0.05, 0.01, 0.95, 0.99, wspace=0, hspace=0.4)
#font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
#wordcloud = WordCloud(font_path)

for i, (gid, s) in enumerate(res_group):
    ax_left, ax_right = axes[i]
    course_name = df.loc[df['id'] == gid, '课程名'].unique()[-1]
    figure.set_label(course_name)

    ax_left.set_title(course_name)
    s.reset_index(level=0, drop=True, inplace=True)
    s.plot.bar(ax=ax_left)

    for label in ax_left.xaxis.get_ticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    img = wordcloud.generate(' '.join(s.index))
    ax_right.set_title('课程标签')
    ax_right.axis("off")
    ax_right.imshow(img, aspect='auto', interpolation='bilinear')

    #img = wordcloud.generate(' '.join(s.index))

figure.savefig('res.png')
