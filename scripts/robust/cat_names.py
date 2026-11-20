"""Hand-written EN/ZH names, synonyms and close hypernyms per COCO category: a func query that contains one of these
names the object instead of describing its function (used by finalize.py)."""
N = {
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
NAMES = {k: v.split(",") for k, v in N.items()}
if __name__ == "__main__":
    import json, sys
    json.dump(NAMES, open(sys.argv[1], "w"), ensure_ascii=False)
    print(len(NAMES))
