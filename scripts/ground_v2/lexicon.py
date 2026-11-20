"""Category lexicon for the v1.2 grounding pipeline: EN/ZH names, synonyms, plurals, similar categories.

SYN is copied from scripts/robust/cat_names.py (S3 robustness sprint, not yet committed when this was written); when
both land on main, import it from there instead of keeping two copies.
"""
import re

SYN = {
 "person": "person,man,woman,people,guy,lady,boy,girl,kid,child,人,男人,女人,男的,女的,小孩,孩子",
 "bicycle": "bicycle,bike,cycle,自行车,单车,脚踏车", "car": "car,automobile,sedan,汽车,轿车,小车",
 "motorcycle": "motorcycle,motorbike,bike,scooter,摩托车,摩托,电动车", "airplane": "airplane,plane,aircraft,jet,airliner,飞机,客机",
 "bus": "bus,coach,公交车,公交,巴士,大巴", "train": "train,locomotive,railcar,火车,列车,地铁,动车",
 "truck": "truck,lorry,pickup,卡车,货车,皮卡", "boat": "boat,ship,vessel,watercraft,yacht,canoe,kayak,船,小船,轮船,舟,艇",
 "traffic light": "traffic light,traffic signal,stoplight,红绿灯,信号灯,交通灯", "fire hydrant": "fire hydrant,hydrant,消防栓",
 "stop sign": "stop sign,停车标志,停止标志,停车牌", "parking meter": "parking meter,meter,停车计时器,咪表,停车收费表",
 "bench": "bench,长椅,长凳,板凳", "bird": "bird,鸟,小鸟,鸽子,鸭", "cat": "cat,kitty,kitten,猫,小猫,猫咪",
 "dog": "dog,puppy,pup,狗,小狗,狗狗", "horse": "horse,pony,马,小马", "sheep": "sheep,lamb,羊,绵羊,小羊",
 "cow": "cow,cattle,bull,ox,calf,牛,奶牛,黄牛", "elephant": "elephant,大象,象", "bear": "bear,熊,狗熊,棕熊",
 "zebra": "zebra,斑马", "giraffe": "giraffe,长颈鹿", "backpack": "backpack,rucksack,knapsack,bag,背包,书包,双肩包,包",
 "umbrella": "umbrella,parasol,伞,雨伞,阳伞", "handbag": "handbag,purse,bag,手提包,手袋,包,钱包,挎包",
 "tie": "tie,necktie,bowtie,领带,领结", "suitcase": "suitcase,luggage,baggage,行李箱,箱子,旅行箱,拉杆箱,行李",
 "frisbee": "frisbee,flying disc,disc,飞盘", "skis": "skis,ski,滑雪板,雪橇", "snowboard": "snowboard,滑雪板,单板",
 "sports ball": "ball,football,soccer ball,baseball,tennis ball,basketball,球,足球,网球,篮球,棒球",
 "kite": "kite,风筝", "baseball bat": "baseball bat,bat,球棒,棒球棒,球棍", "baseball glove": "baseball glove,glove,mitt,手套,棒球手套",
 "skateboard": "skateboard,滑板", "surfboard": "surfboard,board,冲浪板", "tennis racket": "tennis racket,racket,racquet,球拍,网球拍",
 "bottle": "bottle,瓶,瓶子,水瓶,酒瓶", "wine glass": "wine glass,glass,goblet,酒杯,高脚杯,红酒杯,杯",
 "cup": "cup,mug,glass,teacup,tumbler,杯,杯子,茶杯,马克杯,水杯", "fork": "fork,叉,叉子,餐叉",
 "knife": "knife,blade,刀,刀子,餐刀,小刀", "spoon": "spoon,ladle,勺,勺子,汤匙,调羹", "bowl": "bowl,dish,碗,盆,大碗",
 "banana": "banana,香蕉", "apple": "apple,苹果", "sandwich": "sandwich,burger,sub,三明治,汉堡",
 "orange": "orange,tangerine,橙子,橘子,桔子,橙", "broccoli": "broccoli,西兰花,花椰菜,西蓝花", "carrot": "carrot,胡萝卜,萝卜",
 "hot dog": "hot dog,hotdog,sausage,热狗,香肠", "pizza": "pizza,披萨,比萨", "donut": "donut,doughnut,甜甜圈,面包圈",
 "cake": "cake,cupcake,蛋糕", "chair": "chair,seat,stool,椅子,凳子,座椅", "couch": "couch,sofa,loveseat,沙发",
 "potted plant": "potted plant,plant,houseplant,flower pot,盆栽,植物,花盆,盆景", "bed": "bed,mattress,床,床铺",
 "dining table": "dining table,table,desk,桌子,餐桌,桌", "toilet": "toilet,commode,厕所,马桶,坐便器",
 "tv": "tv,television,monitor,screen,电视,电视机,显示器,屏幕", "laptop": "laptop,notebook,computer,笔记本,电脑,笔记本电脑",
 "mouse": "mouse,鼠标", "remote": "remote,remote control,controller,遥控器,遥控", "keyboard": "keyboard,键盘",
 "cell phone": "cell phone,phone,cellphone,smartphone,mobile,手机,电话", "microwave": "microwave,微波炉",
 "oven": "oven,stove,烤箱,炉子,灶", "toaster": "toaster,烤面包机,吐司机", "sink": "sink,basin,水槽,水池,洗手池,洗碗池",
 "refrigerator": "refrigerator,fridge,冰箱", "book": "book,books,书,书本,书籍", "clock": "clock,钟,时钟,挂钟,钟表",
 "vase": "vase,花瓶,瓶", "scissors": "scissors,shears,剪刀,剪子", "teddy bear": "teddy bear,teddy,stuffed animal,plush,泰迪熊,玩偶,毛绒玩具,布娃娃",
 "hair drier": "hair drier,hair dryer,blow dryer,dryer,吹风机,电吹风", "toothbrush": "toothbrush,牙刷",
}
NAMES = {k: v.split(",") for k, v in SYN.items()}
IS_ZH = re.compile(r"[一-鿿]")

# question-facing names: (en singular, en plural, zh, zh measure word). "person" is asked as "person"/"people".
_Q = {
 "person": ("person", "people", "人", "个"), "bicycle": ("bicycle", "bicycles", "自行车", "辆"), "car": ("car", "cars", "汽车", "辆"),
 "motorcycle": ("motorcycle", "motorcycles", "摩托车", "辆"), "airplane": ("airplane", "airplanes", "飞机", "架"),
 "bus": ("bus", "buses", "公交车", "辆"), "train": ("train", "trains", "火车", "列"), "truck": ("truck", "trucks", "卡车", "辆"),
 "boat": ("boat", "boats", "船", "艘"), "traffic light": ("traffic light", "traffic lights", "红绿灯", "个"),
 "fire hydrant": ("fire hydrant", "fire hydrants", "消防栓", "个"), "stop sign": ("stop sign", "stop signs", "停车标志", "个"),
 "parking meter": ("parking meter", "parking meters", "停车计时器", "个"), "bench": ("bench", "benches", "长椅", "张"),
 "bird": ("bird", "birds", "鸟", "只"), "cat": ("cat", "cats", "猫", "只"), "dog": ("dog", "dogs", "狗", "只"),
 "horse": ("horse", "horses", "马", "匹"), "sheep": ("sheep", "sheep", "羊", "只"), "cow": ("cow", "cows", "牛", "头"),
 "elephant": ("elephant", "elephants", "大象", "头"), "bear": ("bear", "bears", "熊", "只"), "zebra": ("zebra", "zebras", "斑马", "匹"),
 "giraffe": ("giraffe", "giraffes", "长颈鹿", "只"), "backpack": ("backpack", "backpacks", "背包", "个"),
 "umbrella": ("umbrella", "umbrellas", "伞", "把"), "handbag": ("handbag", "handbags", "手提包", "个"), "tie": ("tie", "ties", "领带", "条"),
 "suitcase": ("suitcase", "suitcases", "行李箱", "个"), "frisbee": ("frisbee", "frisbees", "飞盘", "个"),
 "skis": ("pair of skis", "pairs of skis", "滑雪板", "副"), "snowboard": ("snowboard", "snowboards", "单板滑雪板", "块"),
 "sports ball": ("ball", "balls", "球", "个"), "kite": ("kite", "kites", "风筝", "只"), "baseball bat": ("baseball bat", "baseball bats", "棒球棒", "根"),
 "baseball glove": ("baseball glove", "baseball gloves", "棒球手套", "只"), "skateboard": ("skateboard", "skateboards", "滑板", "块"),
 "surfboard": ("surfboard", "surfboards", "冲浪板", "块"), "tennis racket": ("tennis racket", "tennis rackets", "网球拍", "个"),
 "bottle": ("bottle", "bottles", "瓶子", "个"), "wine glass": ("wine glass", "wine glasses", "酒杯", "个"), "cup": ("cup", "cups", "杯子", "个"),
 "fork": ("fork", "forks", "叉子", "把"), "knife": ("knife", "knives", "刀", "把"), "spoon": ("spoon", "spoons", "勺子", "把"),
 "bowl": ("bowl", "bowls", "碗", "个"), "banana": ("banana", "bananas", "香蕉", "根"), "apple": ("apple", "apples", "苹果", "个"),
 "sandwich": ("sandwich", "sandwiches", "三明治", "个"), "orange": ("orange", "oranges", "橙子", "个"),
 "broccoli": ("broccoli", "broccoli", "西兰花", "块"), "carrot": ("carrot", "carrots", "胡萝卜", "根"),
 "hot dog": ("hot dog", "hot dogs", "热狗", "个"), "pizza": ("pizza", "pizzas", "披萨", "个"), "donut": ("donut", "donuts", "甜甜圈", "个"),
 "cake": ("cake", "cakes", "蛋糕", "块"), "chair": ("chair", "chairs", "椅子", "把"), "couch": ("couch", "couches", "沙发", "张"),
 "potted plant": ("potted plant", "potted plants", "盆栽", "盆"), "bed": ("bed", "beds", "床", "张"),
 "dining table": ("table", "tables", "桌子", "张"), "toilet": ("toilet", "toilets", "马桶", "个"), "tv": ("TV", "TVs", "电视", "台"),
 "laptop": ("laptop", "laptops", "笔记本电脑", "台"), "mouse": ("computer mouse", "computer mice", "鼠标", "个"),
 "remote": ("remote control", "remote controls", "遥控器", "个"), "keyboard": ("keyboard", "keyboards", "键盘", "个"),
 "cell phone": ("cell phone", "cell phones", "手机", "部"), "microwave": ("microwave", "microwaves", "微波炉", "台"),
 "oven": ("oven", "ovens", "烤箱", "台"), "toaster": ("toaster", "toasters", "烤面包机", "台"), "sink": ("sink", "sinks", "水槽", "个"),
 "refrigerator": ("refrigerator", "refrigerators", "冰箱", "台"), "book": ("book", "books", "书", "本"), "clock": ("clock", "clocks", "钟", "个"),
 "vase": ("vase", "vases", "花瓶", "个"), "scissors": ("pair of scissors", "pairs of scissors", "剪刀", "把"),
 "teddy bear": ("teddy bear", "teddy bears", "泰迪熊", "只"), "hair drier": ("hair dryer", "hair dryers", "吹风机", "个"),
 "toothbrush": ("toothbrush", "toothbrushes", "牙刷", "支"),
}
EN, EN_PL, ZH, ZH_MW = ({k: v[i] for k, v in _Q.items()} for i in range(4))

# visually / semantically close categories (for "similar category" negatives; only used when absent from the image)
SIMILAR = {
 "cat": ["dog", "teddy bear"], "dog": ["cat", "sheep", "bear"], "horse": ["cow", "zebra", "giraffe"], "cow": ["horse", "sheep"],
 "sheep": ["cow", "dog"], "zebra": ["horse", "giraffe"], "giraffe": ["zebra", "horse"], "elephant": ["bear", "cow"],
 "bear": ["dog", "elephant"], "bird": ["kite", "airplane"], "car": ["truck", "bus"], "truck": ["car", "bus", "train"],
 "bus": ["truck", "train", "car"], "train": ["bus", "truck"], "motorcycle": ["bicycle"], "bicycle": ["motorcycle"],
 "airplane": ["bird", "kite", "boat"], "boat": ["airplane", "car"], "cup": ["wine glass", "bowl", "bottle", "vase"],
 "wine glass": ["cup", "bottle", "vase"], "bottle": ["vase", "cup", "wine glass"], "vase": ["bottle", "cup", "potted plant"],
 "bowl": ["cup", "vase"], "fork": ["spoon", "knife"], "spoon": ["fork", "knife"], "knife": ["fork", "spoon", "scissors"],
 "scissors": ["knife"], "chair": ["couch", "bench", "bed"], "couch": ["chair", "bed", "bench"], "bench": ["chair", "couch"],
 "bed": ["couch"], "dining table": ["bed", "bench"], "tv": ["laptop", "microwave"], "laptop": ["tv", "keyboard", "cell phone"],
 "keyboard": ["laptop", "remote"], "remote": ["cell phone", "mouse"], "cell phone": ["remote", "laptop"], "mouse": ["remote", "cell phone"],
 "microwave": ["oven", "toaster", "tv"], "oven": ["microwave", "refrigerator"], "toaster": ["microwave"],
 "refrigerator": ["oven"], "sink": ["toilet", "bowl"], "toilet": ["sink"], "banana": ["carrot", "hot dog"],
 "apple": ["orange"], "orange": ["apple"], "carrot": ["banana"], "broccoli": ["potted plant"], "sandwich": ["hot dog", "pizza", "cake"],
 "hot dog": ["sandwich", "banana"], "pizza": ["cake", "sandwich"], "donut": ["cake", "orange"], "cake": ["pizza", "donut", "sandwich"],
 "skis": ["snowboard"], "snowboard": ["skis", "skateboard", "surfboard"], "skateboard": ["snowboard", "surfboard"],
 "surfboard": ["skateboard", "snowboard", "boat"], "tennis racket": ["baseball bat"], "baseball bat": ["tennis racket"],
 "baseball glove": ["sports ball"], "frisbee": ["sports ball", "kite"], "sports ball": ["frisbee"], "kite": ["bird", "umbrella"],
 "umbrella": ["kite"], "backpack": ["handbag", "suitcase"], "handbag": ["backpack", "suitcase"], "suitcase": ["backpack", "handbag"],
 "tie": ["handbag"], "teddy bear": ["dog", "cat"], "clock": ["tv"], "potted plant": ["vase", "broccoli"],
 "traffic light": ["stop sign", "parking meter"], "stop sign": ["traffic light", "fire hydrant"], "fire hydrant": ["parking meter", "stop sign"],
 "parking meter": ["fire hydrant", "traffic light"], "toothbrush": ["spoon", "hair drier"], "hair drier": ["toothbrush"],
}
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
ZH_NUM = {2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二"}


def mentions(text, words):
    """first of `words` mentioned in text (EN: whole word, optional plural; ZH: substring) or None"""
    t = text.lower()
    for w in words:
        w = w.lower().strip()
        if not w:
            continue
        if IS_ZH.search(w):
            if w in t:
                return w
        elif re.search(r"\b" + re.escape(w) + r"(s|es)?\b", t):
            return w
    return None


def names_of(cat):
    return NAMES.get(cat, [cat]) + [EN[cat], EN_PL[cat], ZH[cat]]


# extra words that count as naming a person (multi-target "mentions every target" check only)
PERSON_WORDS = ("someone,somebody,player,rider,skier,snowboarder,surfer,skateboarder,batter,catcher,umpire,pitcher,"
                "chef,cook,officer,cop,policeman,soldier,worker,biker,cyclist,driver,passenger,tourist,couple,men,women,"
                "children,kids,boys,girls,adult,baby,toddler,teen,teenager,athlete,he,she,him,her,his,they,them,"
                "男子,女子,男孩,女孩,小伙,姑娘,大人,宝宝,婴儿,运动员,球员,骑手,冲浪的,滑雪的,他,她").split(",")


def names_for_mention(cat):
    return names_of(cat) + (PERSON_WORDS if cat == "person" else [])
