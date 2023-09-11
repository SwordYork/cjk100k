# CJK 100K

In this repository, we provide several scripts for analyzing OpenAI's tokenizer as well as comparing it against alternative tokenizers. These scripts help in understanding the performance of the existing tokenizer and can serve as a reference for improving the current tokenizer or creating new ones.

To run the scripts in this repository, you will need to install the following requirements.

```
tiktoken
tokenizers
```


## Analysis of cl100k_base from OpenAI

The [cl100k_base](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) tokenizer is a pre-trained tokenizer from OpenAI, which contains 100,000 tokens (hence the name cl100k_base). It is designed to convert input text into tokens that can be further processed by language models, like GPT-4.

We have parsed the cl100k_base token file and extracted the CJK (Chinese, Japanese, and Korean) tokens. Out of 100,000 tokens, there are only 867 CJK tokens, which represent approximately 0.86% of the total tokens. This relatively small proportion of CJK tokens may not be ideal for efficiently encoding CJK words and sentences. 

```text
{8107: '年', 9039: '数', 9080: '日', 9554: '的', 9953: '月', 11883: '用', 13153: '成', 13372: '名', 13646: '时', 14558: '件', 15225: '请', 16325: '中', 16423: '据', 16882: '码', 16937: '不', 17039: '新', 17161: '文', 17297: '下', 17620: '分', 17701: '入', 17792: '人', 17860: '功', 17905: '上', 17982: '户', 18184: '为', 18363: '间', 18476: '号', 18655: '取', 18904: '回', 19000: '在', 19047: '页', 19113: '字', 19361: '有', 19483: '个', 19653: '成功', 19967: '作', 20379: '示', 20600: '用户', 20675: '数据', 20834: '出', 21043: '是', 21082: '时间', 21388: '失', 21405: '表', 21418: '除', 21601: '加', 21809: '败', 21990: '生', 22023: '信', 22238: '类', 22324: '置', 22649: '理', 22656: '本', 22820: '失败', 23018: '息', 23039: '行', 23187: '定', 23226: '改', 23530: '市', 23538: '期', 23897: '以', 23951: '修', 24186: '元', 24273: '方', 24580: '录', 24775: '区', 24946: '单', 25010: '�除', 25129: '位', 25287: '型', 25333: '法', 25336: '县', 25359: '存', 25446: '品', 25580: '前', 25666: '称', 26062: '�回', 26130: '注', 26239: '修改', 26592: '值', 26794: '输', 26892: '建', 27327: '能', 27384: '大', 27452: '例', 27479: '度', 27704: '始', 27996: '文件', 28037: '到', 28190: '面', 28359: '�数', 28466: '载', 28469: '信息', 28542: '点', 28587: '��取', 28741: '密', 28833: '动', 28873: '果', 29129: '图', 29172: '提', 29391: '发', 29430: '式', 29504: '国', 29706: '删除', 29741: '登', 29826: '错', 30046: '者', 30051: '认', 30156: '误', 30177: '接', 30356: '关', 30358: '重', 30537: '第', 30590: '地', 30624: '如', 30735: '设', 30832: '目', 30867: '开', 30926: '事', 31041: '�数', 31091: '名称', 31540: '可', 31634: '要', 31640: '代', 31809: '小', 31867: '选', 31944: '标', 31958: '明', 31968: '编', 32018: '求', 32218: '列', 32239: '网', 32296: '输入', 32307: '万', 32335: '最', 32438: '�建', 32626: '返回', 32648: '器', 32938: '所', 32943: '内', 33005: '类型', 33014: '体', 33035: '通', 33052: '务', 33091: '此', 33122: '商', 33144: '序', 33200: '错误', 33208: '化', 33420: '消', 33476: '否', 33563: '保', 33655: '使', 33671: '次', 33748: '机', 33764: '对', 33765: '参数', 33857: '量', 33904: '函数', 33967: '密码', 33976: '查', 34048: '部', 34171: '性', 34208: '和', 34226: '更', 34547: '后', 34577: '证', 34972: '题', 35056: '确', 35083: '格', 35287: '了', 35304: '于', 35330: '金', 35417: '公', 35424: '午', 35757: '円', 35818: '片', 35894: '空', 35959: '请求', 36225: '��加', 36343: '态', 36515: '登录', 36651: '管', 36668: '主', 36827: '天', 37026: '自', 37046: '我', 37087: '全', 37271: '今', 37395: '页面', 37507: '来', 37648: '��作', 37656: '正', 37687: '说', 37689: '意', 37705: '送', 37729: '容', 37767: '已', 37985: '结', 38093: '会', 38129: '使用', 38574: '段', 38609: '�认', 38743: '计', 39045: '，请', 39084: '源', 39135: '色', 39177: '時', 39209: '交', 39276: '系', 39282: '过', 39312: '电', 39365: '询', 39404: '符', 39442: '未', 39607: '程', 40053: '常', 40089: '条', 40195: ' 下', 40265: '当', 40452: '管理', 40466: '��态', 40474: '情', 40526: '口', 40862: '合', 41007: '方法', 41053: '车', 41073: '实', 41127: '组', 41190: '操作', 41401: '版', 41642: '周', 41723: '址', 41771: ' 获取', 41914: '记', 41920: '二', 42016: '同', 42052: '业', 42081: '权', 42246: '其', 42399: '进', 42421: '试', 42462: '验', 42506: '料', 42783: '传', 43032: '述', 43167: '集', 43240: '多', 43292: '无', 43323: '员', 43378: '报', 43511: '他', 43568: '無', 43955: '添加', 44309: '服', 44368: '线', 44388: '这', 44416: '制', 44689: ' 的', 44816: '�始', 44820: '�单', 44915: '内容', 45018: '设置', 45059: '生成', 45163: '将', 45191: '状态', 45277: '列表', 45390: '处', 45472: ' 输', 45736: '高', 45829: '子', 45893: '道', 45934: '�述', 46028: '章', 46031: '字段', 46034: '手', 46056: '库', 46091: '三', 46239: '提示', 46281: '从', 46456: '支', 46729: '家', 46885: '日期', 46961: '长', 47000: '付', 47012: '获取', 47018: '秒', 47030: '图片', 47043: '商品', 47095: '路', 47200: '代码', 47406: '完', 47523: '象', 47548: '则', 47551: '现', 47566: ' 设', 47577: '地址', 47585: '保存', 47653: '京', 47770: '转', 47971: '�示', 48039: '辑', 48044: '一个', 48249: '限', 48463: '默认', 48634: '力', 48706: '存在', 48785: ' 数', 48858: ' 创建', 48864: '学', 48915: '外', 48972: '调', 48974: '服务', 48982: '项', 49055: '请输入', 49409: '北', 49491: '字符', 49792: '工', 49838: '笑', 49928: '监', 49988: '任', 50021: '相', 50027: '验证', 50034: '微', 50126: '册', 50182: '联', 50211: '平', 50285: '增', 50287: '听', 50338: '解', 50667: '等', 50928: '得', 51107: '更新', 51109: '收', 51142: ' 用户', 51202: '选�', 51385: '安', 51392: '价', 51431: ' 第', 51450: '取消', 51466: '藏', 51477: '创建', 51504: '选择', 51510: '订单', 51609: '命', 51611: '应', 51747: '为空', 52030: '看', 52084: '索', 52188: '�始化', 52225: '资', 52254: '查询', 52332: '产', 52563: '表示', 52675: '串', 52927: '布', 53229: '原', 53283: '知', 53434: '级', 53610: '水', 53626: '上传', 53802: '监听', 53826: '击', 53901: '好', 53953: '物', 54140: ' 文', 54154: ' 设置', 54253: '不能', 54322: '放', 54456: '亿', 54493: '经', 54581: '描述', 54872: '模', 55030: '之', 55038: '台', 55121: '显示', 55139: '州', 55487: '配', 55642: '处理', 55723: '画', 55758: '统', 55951: ' 是', 55999: '共', 56026: '连', 56235: '海', 56386: '开始', 56438: '所有', 56602: '节', 56716: ' 返回', 56906: '退', 56965: '間', 57106: '比', 57107: '问', 57237: '至', 57378: '备', 57668: '你', 57752: '黑', 58004: ' 下午', 58119: '编辑', 58291: '或', 58318: '与', 58322: '影', 58521: '作者', 58543: '话', 58552: '视', 58653: '读', 58655: '告', 58666: '美', 58721: '事件', 58850: '女', 58911: '山', 59243: ' 和', 59363: ' 生', 59462: '需', 59464: '复', 59505: '手机', 59563: '南', 59614: '必', 59622: '�行', 59757: ' 分', 59795: '中国', 59892: '闭', 59914: '加载', 60174: '城', 60205: '用户名', 60239: '�性', 60251: '结果', 60358: '近', 60455: '效', 60632: '利', 60634: '移', 60843: '总', 60979: '按', 61056: '排', 61075: '首', 61304: '記', 61337: '社', 61496: '标题', 61633: '注意', 61648: '完成', 61710: '确定', 61786: '西', 61826: '先', 61994: '然', 62049: '键', 62205: ' 名', 62249: '周期', 62291: '额', 62543: '写', 62717: '�名', 62789: '注册', 62855: '签', 63091: ' 自', 63212: '因', 63289: '下载', 63344: '如果', 63362: ' 数据', 63397: '命周期', 63679: ' 注', 64022: '别', 64026: '并', 64045: '异', 64063: '束', 64171: ' 修改', 64173: ' 删除', 64179: ' 生命周期', 64209: '心', 64414: '链', 64467: '指', 64479: '评', 64531: '整', 64803: '四', 64889: '断', 64936: '角', 64960: ' 生命周期函数', 65053: '监听页面', 65164: '连接', 65218: ' 上', 65305: '消息', 65372: '软', 65455: '头', 65529: '对象', 65571: '是否', 65573: '邮', 65659: '义', 65743: '司', 65782: '步', 65789: '门', 65820: '导', 65854: '客', 65884: '不能为空', 65917: '右', 66052: '频', 66201: '像', 66378: '特', 66677: '记录', 66776: '非', 66870: '省', 67117: '输出', 67178: '造', 67287: '姓名', 67494: '说明', 67658: '字符串', 67669: '径', 67735: '�试', 67933: '详', 67986: '验证码', 68171: '由', 68379: '包', 68438: '通过', 68464: '东', 68931: '论', 69049: '当前', 69165: '络', 69253: '款', 69272: '�藏', 69362: '支付', 69496: '启', 69636: '而', 69856: '填', 69905: '格式', 69962: '释', 69978: '持', 70041: '��索', 70090: '北京', 70141: '向', 70158: ' 输入', 70203: '算', 70262: ' 对', 70277: '江', 70284: '不存在', 70349: '里', 70453: ' 查', 70472: ' 如', 70525: ' 发', 70542: '份', 70616: '责', 70626: '科', 70694: ' 文件', 70774: ' 类', 70821: '民', 70924: '数组', 71005: '治', 71174: '声', 71208: '男', 71461: '重新', 71600: '设计', 71638: '分类', 71668: ' 输出', 71689: '以上', 71733: '异常', 71869: '族', 71890: '站', 72027: '没', 72069: ' 参数', 72099: '県', 72125: '雅', 72209: '版本', 72234: '换', 72237: '核', 72238: '素', 72368: '都', 72404: '超', 72456: '网络', 72516: '店', 72718: '起', 72794: '隐藏', 72843: '享', 72873: ' 方', 72917: '进行', 73051: ' 是否', 73071: '提交', 73117: '发送', 73164: '联系', 73325: '拉', 73361: '米', 73548: '系统', 73686: '引', 73740: '编号', 73751: '点击', 73769: ' 更', 73958: ' 中', 73981: '语', 74090: '土', 74138: '宋', 74245: '直', 74257: '每', 74318: '公司', 74396: '箱', 74412: ' 字', 74445: '项目', 74482: '後', 74662: ' 在', 74770: '可以', 74843: '参', 75140: '变', 75146: '基', 75259: ' 页面', 75267: '場', 75293: '待', 75320: '程序', 75486: '规', 75493: '数据库', 75513: '政', 75630: '雅黑', 75631: '软雅黑', 75761: '排序', 75863: '也', 75910: '介', 75976: '首页', 76099: '关闭', 76161: '钟', 76208: '五', 76217: '执行', 76323: '审', 76417: '单位', 76455: '手机号', 76502: ' 日', 76505: '木', 76537: '打', 76706: '活', 76718: '微软雅黑', 76750: '播', 76868: '方式', 76982: '该', 77190: ' 初始化', 77195: '条件', 77219: '記事', 77413: '展', 77748: '钮', 77913: '具', 77937: '路径', 78021: '退出', 78111: '宋体', 78228: '志', 78244: '言', 78272: '购', 78388: '但', 78519: '星', 78640: '两', 78657: '例如', 78659: '左', 78698: '考', 78935: '构', 78943: '報', 79059: '球', 79108: '设计器', 79203: ' 更新', 79656: '相关', 79785: '音', 79908: '动生成', 79982: '端', 80003: '，默认', 80019: ' 新', 80073: '搜索', 80172: '投', 80195: '立', 80356: '属性', 80426: '�断', 80578: '们', 80699: '火', 80804: ' 示', 80866: '清', 81194: '金额', 81201: '账', 81258: '就', 81368: '费', 81506: '请选择', 81526: ' 示例', 81543: '没有', 81628: ' 查询', 81646: ' 默认', 81665: '结束', 81742: '案', 81951: ' 控', 81976: ' 请求', 82042: '广', 82267: '确认', 82302: '历', 82317: '及', 82363: ' 如果', 82420: '計', 82533: '止', 82554: ' 方法', 82696: '么', 82768: '货', 82805: '测试', 82900: '数量', 82912: '位置', 82973: '時間', 83042: '�权', 83047: ' 开', 83125: '文章', 83175: '阳', 83266: '队', 83301: '技', 83324: '场', 83337: '链接', 83439: ' 添加', 83639: ' 最', 83687: '数字', 83741: '声明', 83747: '少', 83799: '形', 83800: '产品', 83932: '稿', 83947: '英', 83994: '游', 84095: '亿元', 84131: '分钟', 84410: ' 商', 84844: '供', 84851: '推', 85155: '初始化', 85188: '税', 85284: '按钮', 85663: '無し�', 85707: '初', 85997: ' 当', 86127: '私', 86206: '需要', 86222: ' 解', 86348: '全部', 86354: '景', 86429: '资源', 86436: '去', 86461: '华', 86741: '评论', 86758: ' 使用', 86867: '配置', 87109: ' 不', 87177: '話', 87217: '番', 87219: '问题', 87327: '报道', 87412: '环', 87441: '张', 87447: '開', 87474: '無しさん', 87502: '种', 87646: ' 成', 87844: '易', 88126: '您', 88240: '视频', 88356: '再', 88367: '可能', 88435: '文字', 88631: '板', 88852: '以下', 88905: '电话', 89046: '連', 89151: '真', 89186: '有效', 89408: '今年', 89753: '流', 89783: '余', 89902: '任务', 90070: '见', 90091: '正确', 90112: '给', 90147: '服务器', 90261: '来源', 90354: ' 结', 90756: '详情', 91077: '局', 91082: ' 主', 91272: '优', 91386: '书', 91466: '一页', 91495: '，并', 91547: '发布', 91763: '思', 91774: '見', 91875: '動', 91940: '运', 91951: '审核', 91967: ' 图', 91985: '样', 92019: '其中', 92056: '权限', 92099: '删除成功', 92150: '�新', 92193: '（笑', 92318: ' 时间', 92382: '定义', 92517: ' 关', 92527: ' 登', 92553: '销', 92555: '万元', 92672: '同时', 92693: '無料', 92776: '即', 92780: '只', 92877: '老', 93115: '岁', 93132: '大小', 93233: '找', 93393: ' 实', 93413: ' 或', 93474: '节点', 93598: ' 若', 93636: '小时', 93994: '其他', 94134: '自治', 94249: '分享', 94366: '稍', 94537: '�件', 94588: '达', 94668: '邮箱', 94720: '新增', 94785: ' 提', 94923: '院', 94983: ' 加', 95001: '価', 95221: '気', 95337: '约', 95399: '速', 95475: '停', 95543: '反', 95544: '票', 95598: '十', 96153: '，则', 96356: '身', 96407: ' 商品', 96412: '含', 96455: '率', 96500: '汽', 96511: '专', 96557: '管理员', 97049: '歳', 97150: '，在', 97518: '関', 97522: '议', 97565: '雷', 97655: '正在', 97908: '�能', 98128: ' 自动生成', 98184: '些', 98220: '界', 98245: '陆', 98261: ' 注意', 98390: '备注', 98406: '倍', 98499: '読', 98580: '价格', 98657: '检', 98711: '我的', 98739: '我们', 98806: '还', 98871: '析', 98897: '企', 98915: '友', 99007: '”的', 99337: '简', 99379: '移到', 99397: '問', 99480: '功能', 99496: ' 若要', 99502: '长度', 99741: '装', 99750: '感', 99771: '哈', 99849: '何', 99941: '预', 100066: '送料', 100179: '尔', 100207: '在线'}
```

## CJK 100k

To improve the encoding efficiency for CJK words, you could consider using a pre-trained tokenizer specifically designed for CJK languages. 

We have conducted experiments on several datasets, including RedPajama, OpenWebText, Wikipedia, code and some Chinese datasets, resulting in the creation of the [cjk100k tokenizer](./tokenizer.json). 

The cjk100k tokenizer's coverage is as follows when compared to the cl100k:

```
coverage cjk100k@1000 97.1%
coverage cjk100k@10000 89.86%
coverage cjk100k@100000 63.65%
```

## How to use

We can test or use the cjk100k tokenizer through libraries such as tiktoken or tokenizers. Among them, tiktoken offers better performance.

### tiktoken

Please refer to the [`demo.py`](./demo.py) file, its usage does not significantly differ from the standard usage of the cl100k tokenizer.

```python
def create_tokenizer():
  special_tokens = [
    "[PAD]", "<|endoftext|>", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|beginoftext|>", "<|fim_prefix|>",
    "<|fim_middle|>", "<|fim_suffix|>", "<|beginofprompt|>", "<|endofprompt|>", "[PAD1]", "[PAD2]", "[PAD3]", "[PAD4]",
    "[PAD5]", "[PAD6]", "[PAD7]", "[PAD8]", "[PAD9]", "[PAD10]", "[PAD11]", "[PAD12]", "[PAD13]", "[PAD14]", "[PAD15]",
    "[PAD16]"
  ]

  cjk_regex = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}--[\u4e00-\u9fa5]]+|[\u4e00-\u9fa5]|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?![\S--[\u4e00-\u9fa5]])|\s+"""
  with open(tokenizer_config) as fr:
    ddict = json.loads(fr.read())
    bpe_ranks = data_gym_to_mergeable_bpe_ranks(ddict["model"]["merges"], start_idx=len(special_tokens))
    sp_ranks = {t: i for i, t in enumerate(special_tokens)}
    targs = {"name": "cjk100k", "pat_str": cjk_regex, "mergeable_ranks": bpe_ranks, "special_tokens": sp_ranks}
  return Encoding(**targs)
```


### tokenizers

Please give it a try yourself.


## Details

The main difference between the cjk100k and the cl100k is in the regular expressions used, which results in distinct initial splitting of the raw input strings.

- cl100k，please check the [`pat_str`](https://github.com/openai/tiktoken/blob/ec7c121e385bf1675312c6c33734de6b392890c4/tiktoken_ext/openai_public.py#L76).
```python
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

- cjk100k
```python
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}--[\u4e00-\u9fa5]]+|[\u4e00-\u9fa5]|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?![\S--[\u4e00-\u9fa5]])|\s+
```

There are two main differences in the regular expressions used:

1. Splitting of numerical strings: cl100k splits sequences with a maximum of 3 numeric characters, while cjk100k slices numerical strings into single digits (consistent with the behavior of the llama tokenizer). In terms of calculation, llama's splitting method seems to be more efficient.
2. Handling of CJK characters: cjk100k separates the characters individually in the Unicode range \u4e00-\u9fa5. Although this splitting method may not be the most efficient encoding strategy for CJK characters, it has two advantages:
  a. The encoding efficiency for CJK characters is at least higher than that of cl100k.
  b. It can avoid biases introduced by potential errors in pre-tokenization, like 分词.

## How to Train Your cjk100k Tokenizer 

You can train your cjk100k tokenizer using the tokenizers library, and refer to the following resources for guidance.

1. https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
2. https://huggingface.co/docs/tokenizers/quicktour
3. https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-bpe-tokenizer-from-scratch

In short,

1. prepare your dataset
2. modify [`byte_level_bpe.py`](https://github.com/huggingface/tokenizers/blob/61136666243ab9ebc44f5de96c3caf97c2228e51/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L57) in `huggingface/tokenizers`.


```python
from tokenizers import pre_tokenizers, Tokenizer, Regex, AddedToken

class ByteLevelBPETokenizer(BaseTokenizer):
  
  # ... skip

  cjk_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}&&[^\u4e00-\u9fa5]]+|[\u4e00-\u9fa5]|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?![\S&&[^\u4e00-\u9fa5]])|\s+"

  # https://github.com/huggingface/tokenizers/blob/61136666243ab9ebc44f5de96c3caf97c2228e51/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L57
  # replace pre_tokenizer
  # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
  #     add_prefix_space=add_prefix_space)
  pretok_split_regex = pre_tokenizers.Split(pattern=Regex(cjk_regex),
                                            behavior="isolated")
  pretok_byte = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space,
                                          use_regex=False)
  tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
      [pretok_split_regex, pretok_byte])
```

## TODO

Due to resource constraints, we have not yet fully evaluated the effectiveness of the cjk100k tokenizer. We may perform further validation in the future.