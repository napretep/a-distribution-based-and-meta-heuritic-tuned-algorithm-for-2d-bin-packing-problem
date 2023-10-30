#include <cassert>
#include <iostream>
#include <vector>
#include <variant>
#include <sstream>
#include <random>
using namespace std;
std::string gen_uuid(std::size_t length=8) {
    const std::string CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

    std::string random_string;

    for (std::size_t i = 0; i < length; ++i) {
        random_string += CHARACTERS[distribution(generator)];
    }

    return random_string;
}

class POS {
public:
    double x;
    double y;
    // 默认构造函数
    POS(double x = 0, double y = 0) : x(x), y(y) {}

    // 复制函数
    POS copy() const {
        return POS(this->x, this->y);
    }

    // 减法运算符重载
    POS operator-(const POS& other) const {
        return POS(this->x - other.x, this->y - other.y);
    }
    POS operator-(double scalar) const {
        return POS(this->x - scalar, this->y - scalar);
    }
    // 除法运算符重载
    POS operator/(double scalar) const {
        return POS(this->x / scalar, this->y / scalar);
    }

    // 乘法运算符重载
    POS operator*(double scalar) const {
        return POS(this->x * scalar, this->y * scalar);
    }

    // 加法运算符重载 - POS + POS
    POS operator+(const POS& other) const {
        return POS(this->x + other.x, this->y + other.y);
    }

    // 加法运算符重载 - POS + scalar
    POS operator+(double scalar) const {
        return POS(this->x + scalar, this->y + scalar);
    }
    

    // 等于运算符重载
    bool operator==(const POS& other) const {
        return this->x == other.x && this->y == other.y;
    }

    // 大于运算符重载
    bool operator>(const POS& other) const {
        return this->x > other.x && this->y > other.y;
    }

    // 小于运算符重载
    bool operator<(const POS& other) const {
        return this->x < other.x && this->y < other.y;
    }

    // 大于等于运算符重载
    bool operator>=(const POS& other) const {
        return this->x >= other.x && this->y >= other.y;
    }

    // 小于等于运算符重载
    bool operator<=(const POS& other) const {
        return this->x <= other.x && this->y <= other.y;
    };
    string to_string(){
        std::stringstream ss;
        ss<< "(" << this->x << "," << this->y << ")";
        return ss.str();
    };
};

// 加法运算符重载 - scalar + POS
POS operator+(double scalar, const POS& pos) {
    return POS(pos.x + scalar, pos.y + scalar);
}

enum class TYPE {
    RECT,
    LINE,
    POS
};

class Rect {
    

public:
    POS start;
    POS end;
    int ID;
    // Constructor
    Rect(POS start, POS end, int ID = -1) : start(start), end(end), ID(ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(float x1,float y1,float x2,float y2,int ID = -1):Rect(POS(x1,y1),POS(x2,y2),ID){
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(POS start,float x2,float y2,int ID = -1):Rect(start,POS(x2,y2),ID){
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(float x1,float y1,POS end,int ID = -1):Rect(POS(x1,y1),end,ID){
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(int ID = -1) : start(POS(0,0)), end(POS(0,0)), ID(ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }


    // Operators
    Rect operator-(const POS& other) const{
        return Rect(start - other, end - other, ID);
    }
    Rect operator-(float other) const{
        POS other_new = POS(other,other);
        return Rect(start - other_new, end - other_new, ID);
    }
    Rect operator+(const POS& other) const{
        return Rect(start + other, end + other, ID);
    }
    Rect operator+(float other) const{
        return Rect(start + other, end + other, ID);
    }
    Rect operator*(float other) const{
        return Rect(start * other, end * other, ID);
    }

    // Methods
    POS center() const{
        return (start + end) / 2;
    }
    POS topLeft() const{
        return POS(start.x, end.y);
    }

    POS topRight() const{
        return end;
    }

    POS bottomLeft() const {
        return start;
    }

    POS bottomRight() const{
        return POS(end.x, start.y);
    }

    POS size() const{
        return POS(width(), height());
    }

    double width() const{
        return end.x - start.x;
    }

    double height() const{
        return end.y - start.y;
    }

    double area() const{
        return width() * height();
    }

    Rect transpose() const{
        POS new_end = POS(height(), width()) + start;
        return Rect(start, new_end, ID);
    }

    Rect copy() const{
        return Rect(start, end, ID);
    }

    bool operator==(const Rect& other) const{
        return start == other.start && end == other.end;
    }
    bool operator==(const TYPE& other) const{
        switch (other) {
        case TYPE::RECT:
            return end - start > POS(0,0);
        case TYPE::LINE:
            return ((end.x - start.x == 0 && end.y - start.y > 0) ||
                    (end.x - start.x > 0 && end.y - start.y == 0));
        case TYPE::POS:
            return end - start == POS(0,0);
        default:
            return false;
        }
    }
    bool operator!=(const Rect& other) const {
        return !(*this == other);
    }

    bool contains(const POS& pos) const {
        return pos >= start && pos <= end;
    }
    bool contains(const Rect& rect) const {
        return rect.start >= start && rect.end <= end;
    }
    Rect operator&(const Rect& other) const {
    if (this->contains(other)) {
        return Rect(other.start, other.end);
    }
    else if (other.contains(*this)) {
        return Rect(this->start, this->end);
    }
    else {
        if (this->contains(other.bottomLeft())) {
            if (this->contains(other.bottomRight())) {
                return Rect(other.bottomLeft(), other.bottomRight().x, this->topLeft().y);
            }
            else if (this->contains(other.topLeft())) {
                return Rect(other.bottomLeft(), this->topRight().x, other.topLeft().y);
            }
            else {
                return Rect(other.bottomLeft(), this->topRight());
            }
        }
        else if (this->contains(other.topRight())) {
            if (this->contains(other.topLeft())) {
                return Rect(other.bottomLeft().x, this->bottomLeft().y, other.topRight());
            }
            else if (this->contains(other.bottomRight())) {
                return Rect(this->bottomLeft().x, other.bottomRight().y, other.topRight());
            }
            else {
                return Rect(this->bottomLeft(), other.topRight());
            }
        }
        else if (this->contains(other.topLeft())) {
            if (this->contains(other.topRight())) {
                return Rect(other.bottomLeft().x, this->bottomLeft().y, other.topRight());
            }
            else if (this->contains(other.bottomLeft())) {
                return Rect(other.bottomLeft(), this->topRight().x, other.topLeft().y);
            }
            else {
                return Rect(other.bottomLeft().x, this->bottomLeft().y, this->topRight().x, other.topLeft().y);
            }
        }
        else if (this->contains(other.bottomRight())) {
            if (this->contains(other.bottomLeft())) {
                return Rect(other.bottomLeft(), other.bottomRight().x, this->topLeft().y);
            }
            else if (this->contains(other.topRight())) {
                return Rect(this->bottomLeft().x, other.bottomRight().y, other.topRight());
            }
            else {
                return Rect(this->topLeft().x, other.bottomRight().y, other.bottomRight().x, this->topRight().y);
            }
        }
        else if (this->topLeft().x <= other.topLeft().x && other.topRight().x <= this->topRight().x && 
                 other.bottomRight().y <= this->bottomRight().y && this->topLeft().y <= other.topLeft().y) {
            return Rect(other.bottomLeft().x, this->bottomLeft().y, other.topRight().x, this->topRight().y);
        }
        else if (this->bottomRight().y <= other.bottomRight().y && other.topLeft().y <= this->topLeft().y && 
                 other.topLeft().x <= this->topLeft().x && this->topRight().x <= other.topRight().x) {
            return Rect(this->bottomLeft().x, other.bottomLeft().y, this->topRight().x, other.topRight().y);
        }
        else {
            return Rect();
        }
    }
}

    std::string to_string(){
        std::stringstream ss;
        ss << "Rect(" << start.to_string() << "," << end.to_string()<<")";
        return ss.str();
    }
};

class Container{
    
public:
    Rect rect;
    int plan_id;
    Container(Rect rect, int plan_id=-1){
        this->rect = rect;
        this->plan_id = plan_id;
    };
    
    bool operator==(const Container& other) const{
        return rect == other.rect && plan_id == other.plan_id;
    }
    bool operator==(const Rect& other) const{
        return rect == other;
    }
    bool operator==(const TYPE& other) const{
        return rect == other;
    }
    bool operator!=(const Container& other) const{
        return !(*this == other);
    }
    bool contains(const POS& pos) const {
        return rect.contains(pos);
    }
    bool contains(const Rect& rect) const {
        return this->rect.contains(rect);
    }
    bool contains(const Container& other) const {
        return this->rect.contains(other.rect);
    }

};

class Item{
    
public:
    int ID;
    Rect size;
    POS pos;
    Item(int ID, Rect size, POS pos){
        this->ID = ID;
        this->size = size;
        this->pos = pos;
    }
    
    bool operator==(const Item& other) const{
        return this->ID == other.ID;
    }
    
    bool operator!=(const Item& other) const{
        return !(*this == other);
    }
    
    Rect get_rect(){
        return size + pos;
    }
    
    Item copy(){
        return Item(this->ID, this->size, this->pos);
    }
};

class ProtoPlan{
public:
    int ID;
    Rect material;
    vector<Item> item_sequence;
    vector<Container> remain_containers;
    ProtoPlan(int ID, Rect material, vector<Item> item_sequence, vector<Container> remain_containers){
        this->ID = ID;
        this->material = material;
        this->item_sequence = item_sequence;
        this->remain_containers = remain_containers;
    }
    float get_util_rate()const{
        double total_area = 0.0;
        for (const auto& item : item_sequence) {
            total_area += item.size.area();
        }
        double ratio = total_area / material.area();
        return ratio;
    }

    vector<Container> get_remain_containers(){
        if ( !remain_containers.empty()){
            return remain_containers;
        }
        else{
            throw std::runtime_error("get_remain_containers must be implemented in a derived class when remain_containers is empty");
        }
    }

};

class Algo{
public:
    vector<Item> items;
    Rect material;
    vector<ProtoPlan> solution;
    string task_id;
    Algo(vector<float> flat_items, pair<float,float> material, string task_id=""){
        for (int i = 0; i < flat_items.size(); i += 3){
            this->items.push_back(Item(int(i), Rect(0,0,flat_items[i+1], flat_items[i+2]),POS(0, 0)));
        }
        this->material = Rect(0,0,material.first, material.second);
        this->solution = vector<ProtoPlan>();
        if(task_id != ""){
            this->task_id = task_id;
        }else{
            this->task_id = gen_uuid(8);
        };
        cout<<"task_id: "<<this->task_id<<endl;
    }

    float get_avg_util_rate(){
        double total_rate = 0.0;
        for (const auto& plan : this->solution) {
            total_rate += plan.get_util_rate();
        }
        double ratio = total_rate / this->solution.size();
        return ratio;
    }

};



void test_POS(){
    POS p1(1, 2);
    POS p2(3, 4);
    POS p3 = p1 + p2;
    cout << p3.to_string() << endl;
    cout << (p3+1).to_string() << endl;
    cout << (p3*10).to_string() << endl;
    cout << (p3/10).to_string() << endl;
    cout << (p3>p2)<< endl;
}

void test_rect() {
    std::vector<vector<vector<float>>> data = {
        {{0, 0, 10, 10}, {5, 5, 15, 15}, {5, 5, 10, 10}},
        {{5, 0, 15, 10}, {0, 5, 10, 15}, {5, 5, 10, 10}},
        {{5, 5, 15, 15}, {0, 0, 10, 10}, {5, 5, 10, 10}},
        {{0, 5, 10, 15}, {5, 0, 15, 10}, {5, 5, 10, 10}},
        {{0, 5, 5, 10}, {1, 0, 4, 6}, {1, 5, 4, 6}},
        {{0, 0, 5, 5}, {4, 1, 8, 4}, {4, 1, 5, 4}},
        {{0, 0, 5, 5}, {1, 4, 4, 6}, {1, 4, 4, 5}},
        {{1, 0, 6, 5}, {0, 1, 2, 4}, {1, 1, 2, 4}},
        {{1, 0, 6, 5}, {0, 1, 7, 4}, {1, 1, 6, 4}},
        {{0, 1, 5, 6}, {1, 0, 4, 7}, {1, 1, 4, 6}}
    };

    for (auto& vec : data) {
        Rect A((vec[0][0]), (vec[0][1]), (vec[0][2]), (vec[0][3]),-1);
        Rect B((vec[1][0]), (vec[1][1]), (vec[1][2]), (vec[1][3]),-1);
        Rect C((vec[2][0]), (vec[2][1]), (vec[2][2]), (vec[2][3]),-1);

        Rect D = A & B;

        std::cout << "A=" << A.to_string() << ", B=" << B.to_string()  << ", A&B=" << D.to_string()  << "==C " << (D == C) << std::endl;
    }
    using VarType = std::variant<vector<int>, TYPE>;
    std::vector<std::vector<VarType>> data2 = {
        {std::vector<int>{0, 0, 10, 10}, std::vector<int>{10, 0, 15, 8}, TYPE::LINE},
        {std::vector<int>{0, 0, 10, 10}, std::vector<int>{10, 10, 15, 15}, TYPE::POS},
        {std::vector<int>{5, 5, 15, 15}, std::vector<int>{0, 0, 10, 10}, TYPE::RECT},
    };
    for (auto& vec : data2) {
        auto A_val = get<vector<int>>(vec[0]);
        auto B_val = get<vector<int>>(vec[1]);
        Rect A(A_val[0], A_val[1], A_val[2], A_val[3],-1);
        Rect B(B_val[0], B_val[1], B_val[2], B_val[3],-1);
        // Rect C((vec[2][0]), (vec[2][1]), (vec[2][2]), (vec[2][3]));
        TYPE C = get<TYPE>(vec[2]);
        Rect D = A & B;
        std::cout << "A=" << A.to_string() << ", B=" << B.to_string()  << ", A&B=" << D.to_string()  << "==C " << (D == C) << std::endl;
    }
}

vector<float> test_item_data={
257,308,110,355,308,110,489,308,110,10,308,70,86,308,70,256,308,70,354,308,70,488,308,70,523,308,70,522,350,90,578,350,90,594,350,90,767,350,90,214,346,60,222,346,60,341,346,60,380,346,60,637,346,60,749,400,100,810,400,100,157,400,90,185,400,90,254,400,90,352,400,90,486,400,90,615,400,90,119,400,80,747,398,372,809,398,258,748,398,160,808,398,160,441,396,326,253,396,247,351,396,247,485,396,247,42,782,362,862,779,364,458,778,422,24,774,362,25,774,362,38,774,362,41,774,362,865,773,391,725,773,360,727,773,360,794,773,360,796,773,360,
};


void test_algo(){
    Algo algo(test_item_data, make_pair(1000,1000));
    cout << algo.items.size() << endl;
}

int main(){
    test_algo();
    return 0;
}
