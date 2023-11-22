#include <cassert>
#include <iostream>
#include <vector>
#include <variant>
#include <sstream>
#include <random>
#include <numeric>
#include <typeinfo>
#include <optional>
#include <algorithm>
#include <cmath>
#include <random>
#include <math.h>
//#include <Eigen/Dense>
#include <stdexcept>
using namespace std;

std::string gen_uuid(std::size_t length = 8) {
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

class Rect;
using RectAsVector = vector<float>;
using PlanPartAsVector = vector<RectAsVector>;
using ItemSequenceAsVector = PlanPartAsVector;
using RemainContainersAsVector = PlanPartAsVector;
using PlanAsVector = pair<ItemSequenceAsVector, RemainContainersAsVector>;
using PlanPackingLog = vector<PlanAsVector>;
using SolutionPackingLog = vector<PlanPackingLog>;
using SolutionAsVector = vector<PlanAsVector>;
using RectDivided = vector<optional<Rect>>;

class POS {
public:
    float x;
    float y;
    POS(float x = 0, float y = 0) : x(x), y(y) {}

    POS copy() const {
        return POS(this->x, this->y);
    }
    POS operator-(const POS& other) const {
        return POS(this->x - other.x, this->y - other.y);
    }
    POS operator-(float scalar) const {
        return POS(this->x - scalar, this->y - scalar);
    }
    POS operator/(float scalar) const {
        return POS(this->x / scalar, this->y / scalar);
    }

    POS operator*(float scalar) const {
        return POS(this->x * scalar, this->y * scalar);
    }
    POS operator+(const POS& other) const {
        return POS(this->x + other.x, this->y + other.y);
    }

    POS operator+(float scalar) const {
        return POS(this->x + scalar, this->y + scalar);
    }

    bool operator==(const POS& other) const {
        return this->x == other.x && this->y == other.y;
    }
    bool operator>(const POS& other) const {
        return this->x > other.x && this->y > other.y;
    }
    bool operator<(const POS& other) const {
        return this->x < other.x && this->y < other.y;
    }

    bool operator>=(const POS& other) const {
        return this->x >= other.x && this->y >= other.y;
    }

    bool operator<=(const POS& other) const {
        return this->x <= other.x && this->y <= other.y;
    };
    string to_string()const {
        std::stringstream ss;
        ss << "(" << this->x << "," << this->y << ")";
        return ss.str();
    };
};

POS operator+(float scalar, const POS& pos) {
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
    float maxL, minL, diag;
    int ID;
    // Constructor
    Rect(POS start, POS end, int ID = -1) : start(start), end(end), ID(ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
        //diag = sqrt(pow(width(), 2) + pow(height(), 2));
        maxL = max(width(), height());
        minL = min(width(), height());
    }
    Rect(float x1, float y1, float x2, float y2, int ID = -1) :Rect(POS(x1, y1), POS(x2, y2), ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(POS start, float x2, float y2, int ID = -1) :Rect(start, POS(x2, y2), ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(float x1, float y1, POS end, int ID = -1) :Rect(POS(x1, y1), end, ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    Rect(int ID = -1) : start(POS(0, 0)), end(POS(0, 0)), ID(ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
    }
    vector<float> as_vector() {
        return { start.x, start.y, end.x, end.y };
    }

    // Operators
    Rect operator-(const POS& other) const {
        return Rect(start - other, end - other, ID);
    }
    Rect operator-(float other) const {
        POS other_new = POS(other, other);
        return Rect(start - other_new, end - other_new, ID);
    }
    Rect operator+(const POS& other) const {
        return Rect(start + other, end + other, ID);
    }
    Rect operator+(float other) const {
        return Rect(start + other, end + other, ID);
    }
    Rect operator*(float other) const {
        return Rect(start * other, end * other, ID);
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

    RectDivided operator/(Rect other) const {
        optional<Rect> top, btm, left, right;
        vector<optional<Rect>> v;
        auto master = (*this);
        auto result = master & other;
        if (result == TYPE::RECT) {
            if (result.topRight().y < master.topRight().y) {
                auto top_c = Rect(POS(master.topLeft().x, result.topLeft().y), master.topRight());
                if (top_c == TYPE::RECT) {
                    top = top_c;
                }
            }
            if (result.bottomRight().y > master.bottomRight().y) {
                auto btm_c = Rect(master.bottomLeft(), POS(master.bottomRight().x, result.bottomRight().y));
                if (btm_c == TYPE::RECT) {
                    btm = btm_c;
                }
            }

            if (result.topRight().x < master.topRight().x) {
                auto left_c = Rect(POS(result.bottomRight().x, master.bottomLeft().y), master.topRight());
                if (left_c == TYPE::RECT) {
                    left = left_c;
                }
            }
            if (result.topLeft().x > master.topLeft().x) {
                auto right_c = Rect(master.bottomLeft(), POS(result.topLeft().x, master.topRight().y));
                if (right_c == TYPE::RECT) {
                    right = right_c;
                }
            }
        }
        //if (master.bottomRight().y < other.topRight().y and other.topRight().y < master.topRight().y) {
        //    auto top_c = Rect(POS(master.topLeft().x, other.topLeft().y), master.topRight());
        //    if (top_c == TYPE::RECT) {
        //        top = top_c;
        //    }
        //}
        //if (master.topRight().y > other.bottomRight().y and other.bottomRight().y > master.bottomRight().y) {
        //    auto btm_c = Rect(master.bottomLeft(), POS(master.bottomRight().x, other.bottomRight().y));
        //    if (btm_c == TYPE::RECT) {
        //        btm = btm_c;
        //    }
        //}

        //if (master.topLeft().x < other.topRight().x and other.topRight().x < master.topRight().x) {
        //    auto left_c = Rect(POS(other.bottomRight().x, master.bottomLeft().y), master.topRight());
        //    if (left_c == TYPE::RECT) {
        //        left = left_c;
        //    }
        //}
        //if (master.topRight().x >other.topLeft().x and other.topLeft().x > master.topLeft().x) {
        //    auto right_c = Rect(master.bottomLeft(), POS(other.topLeft().x, master.topRight().y));
        //    if (right_c == TYPE::RECT) {
        //        right = right_c;
        //    }
        //}
        v.push_back(top);
        v.push_back(btm);
        v.push_back(left);
        v.push_back(right);
        return  v;


    }
    Rect operator|(const Rect& other) const {
        return Rect(min(this->start.x, other.start.x), min(this->start.y, other.start.y),
            max(this->end.x, other.end.x), max(this->end.y, other.end.y));
    }

    bool operator==(const Rect& other) const {
        return start == other.start && end == other.end;
    }
    bool operator==(const TYPE& other) const {
        switch (other) {
        case TYPE::RECT:
            return end - start > POS(0, 0);
        case TYPE::LINE:
            return ((end.x - start.x == 0 && end.y - start.y > 0) ||
                (end.x - start.x > 0 && end.y - start.y == 0));
        case TYPE::POS:
            return end - start == POS(0, 0);
        default:
            return false;
        }
    }
    bool operator!=(const Rect& other) const {
        return !(*this == other);
    }
    // Methods
    POS center() const {
        return (start + end) / 2;
    }
    POS topLeft() const {
        return POS(start.x, end.y);
    }

    POS topRight() const {
        return end;
    }

    POS bottomLeft() const {
        return start;
    }

    POS bottomRight() const {
        return POS(end.x, start.y);
    }

    POS size() const {
        return POS(width(), height());
    }

    float width() const {
        return end.x - start.x;
    }

    float height() const {
        return end.y - start.y;
    }

    float area() const {
        return width() * height();
    }

    Rect transpose() const {
        POS new_end = POS(height(), width()) + start;
        return Rect(start, new_end, ID);
    }

    Rect copy() const {
        return Rect(start, end, ID);
    }

    float height_div_width()const {
        return height() / width();
    }

    float aspect_ratio()const {
        return minL / maxL;
    }

    bool contains(const POS& pos) const {
        return pos >= start && pos <= end;
    }
    bool contains(const Rect& rect) const {
        return rect.start >= start && rect.end <= end;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "Rect(" << start.to_string() << "," << end.to_string() << ")";
        return ss.str();
    }
};

class Container {

public:
    Rect rect;
    int plan_id;
    explicit Container(Rect rect, int plan_id = -1) {
        this->rect = rect;
        this->plan_id = plan_id;
    };

    bool operator==(const Container& other) const {
        return rect == other.rect;
    }
    bool operator==(const Rect& other) const {
        return rect == other;
    }
    bool operator==(const TYPE& other) const {
        return rect == other;
    }
    bool operator!=(const Container& other) const {
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
    string to_string()const {
        std::stringstream ss;
        ss << rect.to_string() << ",plan_id=" << plan_id;
        return ss.str();
    }
};


class Item {

public:
    int ID;
    Rect size;
    POS pos;
    Item(int ID, Rect size, POS pos) {
        this->ID = ID;
        this->size = size;
        this->pos = pos;
    }
    bool operator==(const Item& other) const {
        return this->ID == other.ID;
    }
    bool operator!=(const Item& other) const {
        return !(*this == other);
    }
    Rect get_rect() {
        return size + pos;
    }

    Item copy()const {
        return Item(this->ID, this->size, this->pos);
    }
    Item transpose() {
        return Item(this->ID, this->size.transpose(), this->pos);
    }
};

class ItemScore {
public:
    float _score = 0;
    Item item;
    Container container;
    int plan_id;
    ItemScore(Item item, Container container, float score) :item(item), container(container), _score(score) {
        this->plan_id = container.plan_id;
    }
    float getScore()const {
        return _score;
    }
    void setScore(float score) {
        _score = score;
    }
    bool operator<(ItemScore& other)const {
        return this->getScore() < other.getScore();
    }
    bool operator>(ItemScore& other)const {
        return this->getScore() > other.getScore();
    }
    bool operator==(ItemScore& other)const {
        return this->getScore() == other.getScore();
    }
    bool operator!=(ItemScore& other)const {
        return this->getScore() != other.getScore();
    }

};

class ProtoPlan {
public:
    int ID;
    Rect material;
    vector<Item> item_sequence;
    vector<Container> remain_containers;
    ProtoPlan(int ID, Rect material, vector<Item> item_sequence = vector<Item>(), vector<Container> remain_containers = vector<Container>()) {
        this->ID = ID;
        this->material = material;
        this->item_sequence = item_sequence;
        this->remain_containers = remain_containers;
    }
    float get_util_rate()const {
        float total_area = 0.0;
        for (const auto& item : item_sequence) {
            total_area += item.size.area();
        }
        float ratio = total_area / material.area();
        return ratio;
    }

    virtual vector<Container> get_remain_containers(bool throw_error = true) const {
        if (!remain_containers.empty()) {
            return remain_containers;
        }
        else {
            if (throw_error)
            {
                throw std::runtime_error("get_remain_containers must be implemented in a derived class when remain_containers is empty");
            }
            else {
                return remain_containers;
            }
        }
    }
    void print_remain_containers() {
        cout << "plan_id=" << ID << ",remain_containers_count=" << remain_containers.size() << endl;
        for (const auto& container : get_remain_containers()) {
            cout << container.rect.to_string() << endl;
        }
    }
    PlanAsVector toVector(bool throw_runtime_error = true) {
        ItemSequenceAsVector plan_item;
        for (auto item : item_sequence) {
            plan_item.push_back(item.get_rect().as_vector());
        }
        auto containers = get_remain_containers(throw_runtime_error);
        RemainContainersAsVector plan_container;
        for (auto container : containers) {
            plan_container.push_back(container.rect.as_vector());
        }
        PlanAsVector plan_tuple = { plan_item ,plan_container };
        return plan_tuple;
    }

};

class Algo {
public:
    SolutionPackingLog packinglog;
    vector<Item> items;
    Rect material;
    vector<ProtoPlan> solution;
    string task_id;
    bool is_debug;
    float maxL = 0;
    float minL = 99999;
    Algo(vector<float> flat_items, pair<float, float> material, string task_id = "", bool is_debug = false) :is_debug(is_debug) {
        load_items(flat_items);
        this->material = Rect(0, 0, material.first, material.second);
        this->solution = vector<ProtoPlan>();
        if (task_id != "") {
            this->task_id = task_id;
        }
        else {
            this->task_id = gen_uuid(8);
        };
    }

    virtual float get_avg_util_rate() const {
        double total_rate = 0.0;
        for (const auto& plan : this->solution) {
            total_rate += plan.get_util_rate();
        }
        if (this->solution.size() > 0) {
            double ratio = total_rate / this->solution.size();
            return ratio;
        }
        else {
            throw runtime_error("div zero");
        }

    }
    void load_items(vector<float> flat_items) {
        for (int i = 0; i < flat_items.size(); i += 3) {
            auto width = flat_items[i + 1];
            auto height = flat_items[i + 2];
            this->items.push_back(Item(int(i), Rect(0, 0, width, height), POS(0, 0)));
            if (width > maxL) {
                this->maxL = width;
            }
            if (height < minL) {
                this->minL = height;
            }
        }
    }

    virtual SolutionAsVector solution_as_vector() {
        SolutionAsVector result;
        for (auto plan : this->solution) {
            result.push_back(plan.toVector());
        }
        return result;
    }




    virtual void run() {
        throw std::runtime_error("run must be implemented in a derived class");
    }
};




void test_POS() {
    POS p1(1, 2);
    POS p2(3, 4);
    POS p3 = p1 + p2;
    cout << p3.to_string() << endl;
    cout << (p3 + 1).to_string() << endl;
    cout << (p3 * 10).to_string() << endl;
    cout << (p3 / 10).to_string() << endl;
    cout << (p3 > p2) << endl;
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
        Rect A((vec[0][0]), (vec[0][1]), (vec[0][2]), (vec[0][3]), -1);
        Rect B((vec[1][0]), (vec[1][1]), (vec[1][2]), (vec[1][3]), -1);
        Rect C((vec[2][0]), (vec[2][1]), (vec[2][2]), (vec[2][3]), -1);

        Rect D = A & B;

        std::cout << "A=" << A.to_string() << ", B=" << B.to_string() << ", A&B=" << D.to_string() << "==C " << (D == C) << std::endl;
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
        Rect A(A_val[0], A_val[1], A_val[2], A_val[3], -1);
        Rect B(B_val[0], B_val[1], B_val[2], B_val[3], -1);
        // Rect C((vec[2][0]), (vec[2][1]), (vec[2][2]), (vec[2][3]));
        TYPE C = get<TYPE>(vec[2]);
        Rect D = A & B;
        std::cout << "A=" << A.to_string() << ", B=" << B.to_string() << ", A&B=" << D.to_string() << "==C " << (D == C) << std::endl;
    }
    auto r1 = Rect(0.0, 0.0, 1440.0, 1220.0);
    auto r2 = Rect(2200.0, 300.0, 2440.0, 800.0);
    auto r3 = r1 / r2;
    cout << r1.to_string() << "/" << r2.to_string() << "=[top=" << r3[0].has_value() << ",btm=" << r3[1].has_value() << ",left=" << r3[2].has_value() << ",right=" << r3[3].has_value() << endl;
    r1 = Rect(0.0, 0.0, 2440.0, 1220.0);
    r2 = Rect(2200.0, 300.0, 2440.0, 800.0);
    r3 = r1 / r2;
    cout << r1.to_string() << "/" << r2.to_string() << "=[top=" << r3[0].has_value() << ",btm=" << r3[1].has_value() << ",left=" << r3[2].has_value() << ",right=" << r3[3].has_value() << endl;

}

vector<float> test_item_data = {
0,350,90,1,350,90,2,346,60,3,346,60,4,346,60,5,346,60,6,346,60,7,400,100,8,400,100,9,400,90,10,400,90,11,400,90,12,400,90,13,400,90,14,400,90,15,400,80,16,398,372,17,398,258,18,398,160,19,398,160,20,396,326,21,396,247,22,396,247,23,396,247,24,782,362,25,779,364,26,778,422,27,774,362,28,774,362,29,774,362,30,774,362,31,773,391,32,773,360,33,773,360,34,773,360,35,773,360,36,308,110,37,308,110,38,308,110,39,308,70,40,308,70,41,308,70,42,308,70,43,308,70,44,308,70,45,350,90,46,350,90,47,350,90,48,350,90,49,346,60,50,346,60,51,346,60,52,346,60,53,346,60,54,400,100,55,400,100,56,400,90,57,400,90,58,400,90,59,400,90,60,400,90,61,400,90,62,400,80,63,398,372,64,398,258,65,398,160,66,398,160,67,396,326,68,396,247,69,396,247,70,396,247,71,782,362,72,779,364,73,778,422,74,774,362,75,774,362,76,774,362,77,774,362,78,773,391,79,773,360,80,773,360,81,773,360,82,773,360,83,350,90,84,350,90,85,346,60,86,346,60,87,346,60,88,346,60,89,346,60,90,400,100,91,400,100,92,400,90,93,400,90,94,400,90,95,400,90,96,400,90,97,400,90,98,400,90,99,400,90,100,350,90,101,350,90,102,346,60,103,346,60,104,346,60,105,346,60,106,346,60,107,400,100,108,400,100,109,400,90,110,400,90,111,400,90,112,400,90,113,400,90,114,400,90,115,400,80,116,398,372,117,398,258,118,398,160,119,398,160,120,396,326,121,396,247,122,396,247,123,396,247,124,782,362,125,779,364,126,778,422,127,774,362,128,774,362,129,774,362,130,774,362,131,773,391,132,773,360,133,773,360,134,773,360,135,773,360,136,308,110,137,308,110,138,308,110,139,308,70,140,308,70,141,308,70,142,308,70,143,308,70,144,308,70,145,350,90,146,350,90,147,350,90,148,350,90,149,346,60,150,346,60,151,346,60,152,346,60,153,346,60,154,400,100,155,400,100,156,400,90,157,400,90,158,400,90,159,400,90,160,400,90,161,400,90,162,400,80,163,398,372,164,398,258,165,398,160,166,398,160,167,396,326,168,396,247,169,396,247,170,396,247,171,782,362,172,779,364,173,778,422,174,774,362,175,774,362,176,774,362,177,774,362,178,773,391,179,773,360,180,773,360,181,773,360,182,773,360,183,350,90,184,350,90,185,346,60,186,346,60,187,346,60,188,346,60,189,346,60,190,400,100,191,400,100,192,400,90,193,400,90,194,400,90,195,400,90,196,400,90,197,400,90,198,400,90,199,400,90
};
pair<float, float> test_material = make_pair(2440, 1220);

void test_algo() {
    Algo algo(test_item_data, make_pair(1000, 1000));
    cout << algo.items.size() << endl;
}

//-------------------------------Dist algo-------------------------------------------------------------

//class Dist;



class Dist :public Algo { // Dist_Shelf

public:
    class ScoringSys {
    public:

        Dist* algo;
        static const int item_sorting_param_count = 4;
        static const int container_scoring_param_count = 14;
        static const int gap_merging_param_count = 9;
        static const int total_param_count = item_sorting_param_count + container_scoring_param_count + gap_merging_param_count;
        vector<float> parameters;
        

        vector<float> container_scoring_parameters()const {
            vector<float> p(this->parameters.begin(), this->parameters.begin() + container_scoring_param_count);
            return p;
        }
        vector<float>item_sorting_parameters()const {
            vector<float> p(this->parameters.begin() + container_scoring_param_count, this->parameters.begin() + container_scoring_param_count + item_sorting_param_count);
            return p;
        }
        vector<float>gap_merging_parameters()const {
            vector<float> p(this->parameters.begin() + container_scoring_param_count + item_sorting_param_count, this->parameters.end());
            return p;
        }
        /*float item_sorting(float item_width, float item_height)const;
        float pos_scoring(Item item, Container container)const;
        float pos_scoring(float item_width, float item_height, float  container_begin_x, float  container_begin_y, float  container_width, float  container_height, float  plan_id)const;
        float gap_scoring(float current_plan_id, float current_max_len, float current_min_len, Container new_container, Container old_container)const;*/
        float item_sorting(float item_width, float item_height)const {
            vector<float> X = {
                (item_width * item_height) / (this->algo->minL * this->algo->maxL),
                item_height / item_width,
               static_cast<float>((item_width * item_height) / this->algo->material.area()),
                abs(item_width - item_height) / (this->algo->maxL - this->algo->minL)
            };
            // int result = 0;
            auto p = this->item_sorting_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);

        };
        float pos_scoring(Item item, Container container)const {
            Rect cr = container.rect;
            Rect ir = item.size;
            return this->pos_scoring(ir.width(), ir.height(), cr.start.x, cr.start.y, cr.width(), cr.height(), container.plan_id);
        }
        float pos_scoring(float item_width, float item_height, float  container_begin_x, float  container_begin_y, float  container_width, float  container_height, float  plan_id)const {
            vector<float> X = {
                (item_width * item_height) / (this->algo->minL * this->algo->maxL),
                item_height / item_width,
               static_cast<float>((item_width * item_height) / this->algo->material.area()),
                abs(item_width - item_height) / (this->algo->maxL - this->algo->minL),
                this->algo->solution.size() > 0 ? (plan_id + 1) / (this->algo->solution.size()) : 0,
                (item_width * item_height) / (container_width * container_height),
                1 - (item_width * item_height) / (container_width * container_height),
                1 - item_width / container_width,
                1 - item_height / container_height,
                static_cast<float>((container_width * container_height) / this->algo->material.area()),
                static_cast<float>(container_begin_x / this->algo->material.width()),
                static_cast<float>(container_begin_y / this->algo->material.height()),
                static_cast<float>(this->algo->material.height() / this->algo->material.width()),
                plan_id >= 0 ? algo->solution[static_cast<int>(plan_id)].get_util_rate() : 0,
            };

            auto p = this->container_scoring_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
        };
        float gap_scoring(float current_plan_id, float current_max_len, float current_min_len, Container new_container, Container old_container) const {
            auto cutting_rect = Rect(new_container.rect.bottomLeft(), POS(old_container.rect.bottomLeft().x, new_container.rect.topLeft().y));
            vector<float> X = {
                float(this->algo->current_item_idx) / this->algo->items.size(),
                current_plan_id / this->algo->solution.size(),
                cos((new_container.rect.start.x - old_container.rect.start.x) / current_max_len),
                min(new_container.rect.width(),new_container.rect.height()) / max(new_container.rect.width(),new_container.rect.height()),
                min(old_container.rect.width(),old_container.rect.height()) / max(old_container.rect.width(),old_container.rect.height()),
                cos((current_max_len - current_min_len) / this->algo->material.width()),
                1 - new_container.rect.height() / current_max_len,
                cos(cutting_rect.area() / (current_max_len * current_min_len)),
                min(cutting_rect.width(),cutting_rect.height()) / max(cutting_rect.width(),cutting_rect.height())
            };
            auto p = this->gap_merging_parameters();
            return abs(std::inner_product(X.begin(), X.end(), p.begin(), 0.0));
        };
    };

    ScoringSys scoring_sys;
    int current_item_idx;

    Dist(vector<float> items, pair<float, float> material = { 2440,1220 }, string task_id = "", bool is_debug = false) :
        Algo(items, material, task_id, is_debug)
    {
        scoring_sys.algo = this;
        scoring_sys.parameters = vector<float>(ScoringSys::total_param_count, 1);
    };

    void run() {
        this->solution.clear();
        auto sorted_items = this->sorted_items();
        for (auto i = 0; i < sorted_items.size(); i++) {

            this->current_item_idx = i;
            auto current_minL = (*std::min_element(sorted_items.begin() + i, sorted_items.end(), [](const Item& a, const Item& b) {
                return a.size.height() < b.size.height();
                })).size.height();
                auto current_maxL = (*std::max_element(sorted_items.begin() + i, sorted_items.end(), [](const Item& a, const Item& b) {
                    return a.size.width() < b.size.width();
                    })).size.width();
                    Item new_item = sorted_items[i];

                    vector<ItemScore> score_candidates = this->find_possible_item_candidates(new_item);
                    if (score_candidates.size() == 0) {
                        throw std::runtime_error("score_candidates.size() == 0");
                    }

                    ItemScore best_score = *std::max_element(score_candidates.begin(), score_candidates.end(), [](const ItemScore& p1, const ItemScore& p2) {
                        return p1.getScore() < p2.getScore();
                        });
                    auto best_item = best_score.item;
                    auto new_rect = best_item.get_rect();
                    auto remove_rect = best_score.container.rect;

                    //std::optional<Container> new_BR_corner = Container(Rect(new_rect.bottomRight(), POS(remove_rect.topRight().x, new_rect.topRight().y)), best_score.plan_id);
                    //std::optional<Container> new_top_corner = Container(Rect(new_rect.topLeft(), remove_rect.topRight()), best_score.plan_id);
                    std::optional<Container> new_BR_corner = Container(Rect(new_rect.bottomRight(), remove_rect.topRight()), best_score.plan_id);
                    std::optional<Container> new_top_corner = Container(Rect(new_rect.topLeft(), POS(new_rect.bottomRight().x, remove_rect.topRight().y)), best_score.plan_id);

                    //if ((new_BR_corner.value().rect.end.y - new_BR_corner.value().rect.start.y) < current_minL) {
                    //    new_BR_corner.reset();
                    //}
                    //if ((new_top_corner.value().rect.end.y - new_top_corner.value().rect.start.y) < current_minL) {
                    //    if (new_BR_corner.has_value()) {
                    //        new_BR_corner.value().rect.end = new_top_corner.value().rect.end;
                    //        new_top_corner.reset();
                    //    }
                    //}
                    if (best_score.plan_id == -1) {
                        auto new_plan = ProtoPlan(this->solution.size(), this->material.copy(), vector<Item>{}, vector<Container>{});
                        new_plan.item_sequence.push_back(best_item);
                        if (new_BR_corner.has_value()) {
                            new_BR_corner.value().plan_id = new_plan.ID;
                            new_plan.remain_containers.push_back(new_BR_corner.value());
                        }
                        if (new_top_corner.has_value()) {
                            new_top_corner.value().plan_id = new_plan.ID;
                            new_plan.remain_containers.push_back(new_top_corner.value());
                        }
                        this->solution.push_back(new_plan);
                        if (is_debug) {
                            PlanPackingLog plan_packing_log;
                            plan_packing_log.push_back(new_plan.toVector());
                            this->packinglog.push_back(plan_packing_log);
                        }
                        
                    }
                    else {
                        auto& plan = this->solution[best_score.plan_id];
                        plan.item_sequence.push_back(best_score.item);
                        std::erase(plan.remain_containers, best_score.container);

                        for (auto& container : plan.remain_containers) {
                            if (new_BR_corner.has_value() || new_top_corner.has_value()) {
                                if (new_BR_corner.has_value()) {
                                    new_BR_corner = this->container_merge_thinking(current_minL, current_maxL, plan, new_BR_corner, container);
                                }
                                if (new_top_corner.has_value()) {
                                    new_top_corner = this->container_merge_thinking(current_minL, current_maxL, plan, new_top_corner, container);
                                }
                            }
                            else {
                                break;
                            }
                        }
                        if (new_BR_corner.has_value()) {
                            plan.remain_containers.push_back(new_BR_corner.value());
                        }
                        if (new_top_corner.has_value()) {
                            plan.remain_containers.push_back(new_top_corner.value());
                        }


                        if (is_debug) {
                            auto plancopy = plan;
                            this->packinglog.at(best_score.plan_id).push_back(plancopy.toVector());
                        }
                        
                    }
        }

    }

    vector<ItemScore> find_possible_item_candidates(Item& item) const {
        vector<ItemScore> score_list = {};
        for (const auto& plan : this->solution) {
            auto containers = plan.remain_containers;
            for (const auto& container : containers) {
                auto corner_start = container.rect.start;
                auto corner_end = container.rect.end;
                if (container.contains(item.size + corner_start)) {
                    auto item_to_append = item.copy();
                    item_to_append.pos = corner_start.copy();
                    float score = scoring_sys.pos_scoring(item_to_append, container);
                    score_list.push_back(ItemScore(item_to_append, container, score));
                }
                auto itemT = item.transpose();
                if (container.contains(itemT.size + corner_start)) {
                    auto item_to_append = itemT.copy();
                    item_to_append.pos = corner_start.copy();
                    float score = scoring_sys.pos_scoring(item_to_append, container);
                    score_list.push_back(ItemScore(item_to_append, container, score));
                }
            }
        }
        if (score_list.size() == 0) {
            Container container = Container(material.copy());
            auto corner_start = container.rect.start;
            auto corner_end = container.rect.end;
            if (container.contains(item.size + corner_start)) {
                auto item_to_append = item.copy();
                item_to_append.pos = corner_start;
                float score = scoring_sys.pos_scoring(item_to_append, container);
                score_list.push_back(ItemScore(item_to_append, container, score));
            }
            auto itemT = item.transpose();
            if (container.contains(itemT.size + corner_start)) {
                auto item_to_append = itemT.copy();
                item_to_append.pos = corner_start;
                float score = scoring_sys.pos_scoring(item_to_append, container);
                score_list.push_back(ItemScore(item_to_append, container, score));
            }
        }
        return score_list;
    }

    vector<Item> sorted_items()const {
        std::vector<Item> temp_sorted_items = items;

        std::sort(temp_sorted_items.begin(), temp_sorted_items.end(),
            [this](const Item& a, const Item& b) {
                return scoring_sys.item_sorting(a.size.width(), a.size.height()) < scoring_sys.item_sorting(b.size.width(), b.size.height());
            });

        return temp_sorted_items;
    };


    std::optional<Container> container_merge_thinking(float current_minL, float current_maxL, ProtoPlan& plan, optional<Container>& optional_compare_corner, Container& container) {
        auto compare_corner = optional_compare_corner.value();
        auto result = container.rect & compare_corner.rect;
        bool merged = false;
        float merge_gap = scoring_sys.gap_scoring(float(plan.ID), current_maxL, current_minL, compare_corner, container);
        if (result == TYPE::LINE) {
            auto diff = result.end - result.start;
            if (diff.x == 0) {
                if (compare_corner.rect.bottomLeft() == container.rect.bottomRight() && compare_corner.rect.topLeft() == container.rect.topRight()) {
                    container.rect.end = compare_corner.rect.end.copy();
                    merged = true;
                }
                else if (compare_corner.rect.bottomRight() == container.rect.bottomLeft() && compare_corner.rect.topRight() == container.rect.topLeft()) {
                    container.rect.start = compare_corner.rect.start.copy();
                    merged = true;
                }
            }
            else if (diff.y == 0) {
                if (compare_corner.rect.bottomRight() == container.rect.topRight() && compare_corner.rect.bottomLeft() == container.rect.topLeft()) {
                    container.rect.end = compare_corner.rect.end.copy();
                    merged = true;
                }
                else if (compare_corner.rect.topRight() == container.rect.bottomRight() && compare_corner.rect.topLeft() == container.rect.bottomLeft()) {
                    container.rect.start = compare_corner.rect.start.copy();
                    merged = true;
                }
                else if (compare_corner.rect.topRight() == container.rect.bottomRight() &&
                    std::abs(compare_corner.rect.topLeft().x - container.rect.bottomLeft().x) < merge_gap) {
                    container.rect.start = POS(std::max(container.rect.start.x, compare_corner.rect.start.x), compare_corner.rect.start.y);
                    merged = true;
                }
                else if (compare_corner.rect.bottomRight() == container.rect.topRight() &&
                    std::abs(compare_corner.rect.bottomLeft().x - container.rect.topLeft().x) < merge_gap)
                {
                    container.rect.end = compare_corner.rect.end.copy();
                    container.rect.start = POS(std::max(container.rect.topLeft().x, compare_corner.rect.bottomLeft().x), container.rect.start.y);
                    merged = true;
                }
            }
        }
        return merged ? std::nullopt : std::optional<Container>(compare_corner);
    }
};




class Dist2 :public Algo { //Dist_MaxRect

public:
    class ScoringSys {
    public:
        Dist2* parent;
        vector<float>parameters;
        struct ParameterCount {
            static constexpr auto pos_scoring = 28;
            static constexpr auto sort_scoring = 16;
            static constexpr auto total = pos_scoring + sort_scoring;
        };
        vector<float> get_item_sorting_parameters()const {
            vector<float> p(this->parameters.begin(), this->parameters.begin() + ParameterCount::sort_scoring);
            return p;
        };
        vector<float> get_container_scoring_parameters()const {
            vector<float> p(this->parameters.begin() + ParameterCount::sort_scoring, this->parameters.end());
            return p;
        }

        float item_sorting(float item_width, float item_height)const {
            vector<float> X = {
                    
                    1 - (item_width * item_height) / (parent->minL * parent->maxL),
                    1 - abs(item_width - item_height) / (parent->maxL - parent->minL),
                    1 - (item_width * item_height) / parent->material.area(),
                    1 - item_height / item_width,
                    1 - item_height / parent->maxL,
                    1 - item_width / parent->minL,
                    1 - item_width / parent->material.width(),
                    1 - item_height / parent->material.height(),
            };
            auto X_len = X.size();
            for (auto i = 0; i < X_len; i++) {
                X.push_back(1 - X.at(i));
            }

            auto p = get_item_sorting_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
        }
        float pos_scoring(Item item, Container container, int plan_id)const {
            auto item_rect = item.get_rect();
            auto container_rect = container.rect;
            auto container_remain_top_rect = Rect(item_rect.topLeft(), container_rect.topRight());
            auto container_remain_right_rect = Rect(item_rect.bottomRight(), container_rect.topRight());
            auto container_remain_diag_rect = Rect(item_rect.end, container_rect.end);
            vector<float> X = {
                
                   1 - item_rect.area() / container_rect.area(),
                   1 - this->parent->current_maxL / container_rect.width(),
                   1 - this->parent->current_minL / container_rect.height(),
                   1 - item_rect.width() / container_rect.width(),
                   1 - item_rect.height() / container_rect.height(),
                   1 - item_rect.width() / parent->material.width(),
                   1 - item_rect.height() / parent->material.height(),
                   1 - item_rect.area() / parent->material.area(),
                   1 - container_rect.area() / parent->material.area(),
                   1 - container_rect.start.x / parent->material.width(),
                   1 - container_rect.start.y / parent->material.height(),
                   1 - container_remain_top_rect.area()/container_rect.area(),
                   1 - container_remain_right_rect.area() / container_rect.area(),
                   1 - container_remain_diag_rect.area() / container_rect.area(),
            };
            auto X_len = X.size();
            for (auto i = 0; i < X_len; i++) {
                X.push_back(1 - X.at(i));
            }



            auto p = get_container_scoring_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
        }

    };
    ScoringSys scoring_sys;
    float maxL, minL, current_maxL, current_minL;
    Dist2(vector<float> items, pair<float, float> material = test_material, string task_id = "", bool is_debug = false) :
        Algo(items, material, task_id, is_debug) {
        scoring_sys.parameters = vector<float>(ScoringSys::ParameterCount::pos_scoring + ScoringSys::ParameterCount::sort_scoring, 1);
        scoring_sys.parent = this;
        maxL = (*max_element(this->items.begin(), this->items.end(), [](const Item a, const Item b) {
            return a.size.width() < b.size.width();
            })).size.width();
        minL = (*min_element(this->items.begin(), this->items.end(), [](const Item a, const Item b) {
                return a.size.height() < b.size.height();
                })).size.height();
        current_maxL = maxL;
        current_minL = minL;

    }
    Dist2(vector<float> items, pair<float, float> material = test_material, bool is_debug = false) :Dist2(items, material, "", is_debug) {};

    vector<Item> sorted_items()const {
        std::vector<Item> temp_sorted_items = items;

        std::sort(temp_sorted_items.begin(), temp_sorted_items.end(),
            [this](const Item& a, const Item& b) {
                //descend
                return scoring_sys.item_sorting(a.size.width(), a.size.height()) > scoring_sys.item_sorting(b.size.width(), b.size.height());
            });

        return temp_sorted_items;
    }
    int current_item_idx;
    vector<ItemScore> find_possible_item_candidates(Item item) {
        vector<ItemScore> scores_li;
        // from existing containers
        for (ProtoPlan& plan : solution)
        {

            for (auto& container : plan.remain_containers)
            {
                Item item_prepare = item.copy();
                checkAndScoreItem(item_prepare, container, scores_li);
                Item item_prepareT = item.copy().transpose();
                checkAndScoreItem(item_prepareT, container, scores_li);
            }
        }
        Container fake_container = Container(material.copy());
        Item item_prepare = item.copy();
        checkAndScoreItem(item_prepare, fake_container, scores_li);
        Item item_prepareT = item.transpose();
        checkAndScoreItem(item_prepareT, fake_container, scores_li);

        if (scores_li.size() == 0) {
            throw runtime_error("no possible item candidates");
        }
        return scores_li;
    }

    void checkAndScoreItem(Item item, Container& container, std::vector<ItemScore>& score_list) {        
        auto btmleft_pos = container.rect.start;
        if (container.contains(item.size + btmleft_pos)) {
            auto item_to_append = item.copy();
            item_to_append.pos = container.rect.start.copy();
            float score = scoring_sys.pos_scoring(item_to_append, container, container.plan_id);
            score_list.push_back(ItemScore(item_to_append, container, score));
        }
    }
    void run() {

        this->solution.clear();
        auto sorted_items = this->sorted_items();
        for (auto i = 0; i < sorted_items.size(); i++) {

            this->current_item_idx = i;
            //auto current_minL = (*std::min_element(sorted_items.begin() + i, sorted_items.end(), [](const Item& a, const Item& b) {
            //    return a.size.height() < b.size.height();
            //    })).size.height();
            //auto current_maxL = (*std::max_element(sorted_items.begin() + i, sorted_items.end(), [](const Item& a, const Item& b) {
            //    return a.size.width() < b.size.width();
            //    })).size.width();
                    Item new_item = sorted_items[i];

                    vector<ItemScore> scores_li = this->find_possible_item_candidates(new_item);
                    if (scores_li.size() == 0) {
                        throw std::runtime_error("score_candidates.size() == 0");
                    }

                    auto best_score = *std::min_element(scores_li.begin(), scores_li.end(), [](const ItemScore& p1, const ItemScore& p2) {
                        return p1.getScore() < p2.getScore();
                        });
                    

                    if (best_score.plan_id == -1) {
                        auto plan = ProtoPlan(solution.size(), material);
                        plan.item_sequence.push_back(best_score.item);
                        auto item_rect = best_score.item.get_rect();
                        auto container_top = Container(Rect(item_rect.topLeft(), material.topRight()), plan.ID);
                        auto container_btm = Container(Rect(item_rect.bottomRight(), material.topRight()), plan.ID);
                        plan.remain_containers.push_back(container_top);
                        plan.remain_containers.push_back(container_btm);
                        solution.push_back(plan);

                        if (is_debug) {
                            PlanPackingLog planlog;
                            planlog.push_back(plan.toVector());
                            this->packinglog.push_back(planlog);
                        }
                    }
                    else {
                        auto& plan = solution.at(best_score.plan_id);
                        plan.item_sequence.push_back(best_score.item);
                        auto new_item = best_score.item;
                        auto container = best_score.container;
                        auto new_rect = new_item.get_rect();
                        update_remain_containers(new_item, plan);
                        if (is_debug) {
                            this->packinglog.at(best_score.plan_id).push_back(plan.toVector());
                        }
                    }

        }

    }
    void update_remain_containers(Item& item, ProtoPlan& plan) {
        vector<Container>container_to_remove;
        vector<optional<Container>>container_to_append;

        auto item_rect = item.get_rect();
        // split 
        for (auto& container : plan.remain_containers) {
            /*if ((container.rect & item_rect) != TYPE::RECT) {
                continue;
            }*/
            RectDivided result_rects = container.rect / item_rect;
            bool remove = false;
            for (auto& r : result_rects) {
                if (r.has_value()) {
                    remove = true;
                    container_to_append.push_back(Container(r.value(), plan.ID));
                }
            }
            if (remove) {
                container_to_remove.push_back(container);
            }

        }
        // remove contained rect from new splited rect
        for (auto& containerA : container_to_append) {
            if (!containerA.has_value()) {
                continue;
            }
            for (auto& containerB : container_to_append) {
                if (!containerB.has_value()) {
                    continue;
                }
                if (containerA.value() == containerB.value()) {
                    continue;
                }
                if (containerA.value().contains(containerB.value())) {
                    containerB.reset();
                }
                if (containerB.has_value() and containerB.value().contains(containerA.value())) {
                    containerA.reset();
                    break;
                }
            }
        }
        // remove splited rect from old container
        for (auto& container : container_to_remove) {
            std::erase(plan.remain_containers, container);
            if (is_debug) {
                auto plancopy = plan;
                this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
            }
        }

        // check if new rect is contained
        for (auto& new_container : container_to_append) {
            if (!new_container.has_value()) {
                continue;
            }
            for (auto& old_container : plan.remain_containers) {
                if (old_container.rect.contains(new_container.value().rect)) {
                    new_container.reset();
                    break;
                }
            }
            if (new_container.has_value()) {
                plan.remain_containers.push_back(new_container.value());
                if (is_debug) {
                    auto plancopy = plan;
                    this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
                }
            }
        }


    }
};





class Dist3 :public Algo { // Dist_Skyline
public:
    class ScoringSys {
    public:
        Dist3* parent;
        vector<float>parameters;
        struct ParameterCount {
            static constexpr auto skyline_pos_scoring =28;
            static constexpr auto wastemap_pos_scoring = 28;
            static constexpr auto sort_scoring = 16;
            static constexpr auto total = sort_scoring + wastemap_pos_scoring + skyline_pos_scoring;
        };
        vector<float> get_item_sorting_parameters()const {
            vector<float> p(this->parameters.begin(), this->parameters.begin() + ParameterCount::sort_scoring);
            return p;
        };
        vector<float> get_skyline_pos_scoring_parameters()const {
            vector<float> p(this->parameters.begin() + ParameterCount::sort_scoring, this->parameters.begin() + ParameterCount::sort_scoring + ParameterCount::skyline_pos_scoring);
            return p;
        }
        vector<float> get_wastemap_pos_scoring_parameters()const {
            vector<float> p(this->parameters.begin() + ParameterCount::sort_scoring + ParameterCount::skyline_pos_scoring, this->parameters.end());
            return p;
        }

        float item_sorting(float item_width, float item_height)const {
            vector<float> X = {
                    1 - (item_width * item_height) / (parent->minL * parent->maxL),
                    1 - abs(item_width - item_height) / (parent->maxL - parent->minL),
                    1 - (item_width * item_height) / parent->material.area(),
                    1 - item_height / item_width,
                    1 - item_height / parent->maxL,
                    1 - item_width / parent->minL,
                    1 - item_width / parent->material.width(),
                    1 - item_height / parent->material.height(),
            };
            auto X_len = X.size();
            for (auto i = 0; i < X_len; i++) {
                X.push_back(1 - X.at(i));
            }
            auto p = get_item_sorting_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
        }
        float skyline_container_pos_scoring(Item& item, int begin_idx, int end_idx, vector<Container>& containers, int plan_id)const {
            // when meet this situation , we need to consider the skyline container ,and the wastemap container both
            auto item_rect = item.get_rect();
            auto begin_container_rect = containers.at(begin_idx).rect;
            auto end_container_rect = containers.at(end_idx).rect;
            auto container_remain_top_rect = Rect(item_rect.topLeft(), end_container_rect.topRight());
            auto container_remain_right_rect = Rect(item_rect.bottomRight(), end_container_rect.topRight());
            auto container_remain_diag_rect = Rect(item_rect.end, end_container_rect.end);
            float min_y = containers.at(begin_idx).rect.start.y;
            float waste_area = accumulate(containers.begin(), containers.end(), 0, [min_y](float current_sum, Container b) {
                return current_sum + (min_y - b.rect.start.y) * b.rect.width();
                }
            );
            Rect ideal_skyline_rect = Rect(item_rect.start, end_container_rect.end);
            Rect ideal_waste_rect = containers.size()>1? accumulate(containers.begin()+1, containers.end(), Rect(), [min_y](Rect current_sum, Container b) {
                return current_sum |Rect(b.rect.start,POS(b.rect.end.x, min_y));
                }
            ): Rect();
            float container_total_len = accumulate(containers.begin(), containers.end(), 0, [min_y](float current_sum, Container b) {
                return current_sum + b.rect.width();
                }
            );

            vector<float> X = {
                   1 - (ideal_waste_rect==TYPE::RECT ? waste_area /ideal_waste_rect.area(): 0),
                   1 - this->parent->current_maxL / ideal_skyline_rect.width(),
                   1 - this->parent->current_minL / ideal_skyline_rect.height(),
                   1 - item_rect.width() / ideal_skyline_rect.width(),
                   1 - item_rect.height() / ideal_skyline_rect.height(),
                   1 - item_rect.width() / parent->material.width(),
                   1 - item_rect.height() / parent->material.height(),
                   1 - item_rect.area() / parent->material.area(),
                   1 - ideal_skyline_rect.area() / parent->material.area(),
                   1 - begin_container_rect.start.x / parent->material.width(),
                   1 - begin_container_rect.start.y / parent->material.height(),
                   1 - container_remain_top_rect.area() / ideal_skyline_rect.area(),
                   1 - container_remain_right_rect.area() / ideal_skyline_rect.area(),
                   1 - container_remain_diag_rect.area() / ideal_skyline_rect.area(),
            };
            auto X_len = X.size();
            for (auto i = 0; i < X_len; i++) {
                X.push_back(1 - X.at(i));
            }

            auto p = get_skyline_pos_scoring_parameters();
            auto score =  std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
            return score;
        }

        float wastemap_container_pos_scoring(Item item, Container container, int plan_id)const {
            auto item_rect = item.get_rect();
            auto container_rect = container.rect;
            auto container_remain_top_rect = Rect(item_rect.topLeft(), container_rect.topRight());
            auto container_remain_right_rect = Rect(item_rect.bottomRight(), container_rect.topRight());
            auto container_remain_diag_rect = Rect(item_rect.end, container_rect.end);
            vector<float> X = {

                   1 - item_rect.area() / container_rect.area(),
                   1 - this->parent->current_maxL / container_rect.width(),
                   1 - this->parent->current_minL / container_rect.height(),
                   1 - item_rect.width() / container_rect.width(),
                   1 - item_rect.height() / container_rect.height(),
                   1 - item_rect.width() / parent->material.width(),
                   1 - item_rect.height() / parent->material.height(),
                   1 - item_rect.area() / parent->material.area(),
                   1 - container_rect.area() / parent->material.area(),
                   1 - container_rect.start.x / parent->material.width(),
                   1 - container_rect.start.y / parent->material.height(),
                   1 - container_remain_top_rect.area() / container_rect.area(),
                   1 - container_remain_right_rect.area() / container_rect.area(),
                   1 - container_remain_diag_rect.area() / container_rect.area(),
            };
            auto X_len = X.size();
            for (auto i = 0; i < X_len; i++) {
                X.push_back(1 - X.at(i));
            }



            auto p = get_wastemap_pos_scoring_parameters();
            return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
        }

    };

    class Plan :public ProtoPlan {
    public:

        vector<Container> SkylineContainers;
        vector<Container> WasteMap;
        Plan(int ID, Rect material, vector<Item> item_sequence = vector<Item>(), vector<Container> remain_containers = vector<Container>()) :
            ProtoPlan(ID, material) {};
        vector<Container> get_remain_containers(bool throw_error = true)const override {
            vector<Container> result;
            for (auto c : SkylineContainers) {
                result.push_back(c);
            }
            for (auto c : WasteMap) {
                result.push_back(c);
            }
            return result;
        }

    };
    enum class ContainerType {
        WasteMap,
        Skyline
    };
    string output_containertype(ContainerType type) {
        if (type == ContainerType::WasteMap) {
            return "WasteMap";
        }
        else if (type == ContainerType::Skyline) {
            return "Skyline";
        }
        else {
            throw runtime_error("unkown type");
        }
    }
    class Score {
    public:
        Item item;
        pair<int, int> container_range;
        float score;
        int plan_id;
        Dist3::ContainerType type;
        Score(Item item, pair<int, int> container_range, int plan_id, Dist3::ContainerType type, float score = 0, pair<float, float>scores = { 0,0 }) :item(item), container_range(container_range), score(score), plan_id(plan_id), type(type) {};
    };

    
    ScoringSys scoring_sys;
    float current_minL;
    float current_maxL;
    int run_count = 0;
    vector<Dist3::Plan>solution;
    Dist3(vector<float> flat_items, pair<float, float> material = test_material, string task_id = "", bool is_debug = false) :Algo(flat_items, material, task_id, is_debug) {
        maxL = (*max_element(this->items.begin(), this->items.end(), [](const Item a, const Item b) {
            return a.size.width() < b.size.width();
            })).size.width();
        minL = (*min_element(this->items.begin(), this->items.end(), [](const Item a, const Item b) {
            return a.size.height() < b.size.height();
            })).size.height();
        current_maxL = maxL;
        current_minL = minL;
        scoring_sys.parameters = vector<float>(ScoringSys::ParameterCount::total, 1);
        scoring_sys.parent = this;
    }
    Dist3(vector<float> items, pair<float, float> material = test_material, bool is_debug = false) :Dist3(items, material, "", is_debug) {};

    void get_wastemap_scores(Item& new_item, vector<Dist3::Score>& scores) {
        for (auto& plan : solution) {

            for (auto i = 0; i < plan.WasteMap.size(); i++) {
                Rect waste_rect = plan.WasteMap.at(i).rect;
                if (waste_rect.contains(new_item.size + waste_rect.start)) {
                    auto item_to_append = new_item.copy();
                    item_to_append.pos = waste_rect.start.copy();
                    auto score = Score(item_to_append, { i,i + 1 }, plan.ID, ContainerType::WasteMap, calc_wastemap_score(item_to_append, plan.WasteMap.at(i), plan.ID));
                    scores.push_back(score);
                }
                auto itemT = new_item.transpose();
                if (waste_rect.contains(itemT.size + waste_rect.start)) {
                    auto item_to_append = itemT.copy();
                    item_to_append.pos = waste_rect.start.copy();
                    auto score = Score(item_to_append, { i,i + 1 }, plan.ID, ContainerType::WasteMap, calc_wastemap_score(item_to_append, plan.WasteMap.at(i), plan.ID));
                    scores.push_back(score);
                }
            }

        }
    }
    void get_skyline_scores(Item& new_item, vector<Dist3::Score>& scores) {
        for (auto& plan : solution) {
            for (auto i = 0; i < plan.SkylineContainers.size(); i++) {
                Rect skyline_rect = plan.SkylineContainers.at(i).rect;
                int idx = get_placable_area(new_item, i, plan.SkylineContainers);
                if (idx >= 0) {
                    auto item_to_append = new_item.copy();
                    item_to_append.pos = skyline_rect.start.copy();
                    float score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers, plan.ID);
                    auto score = Score(item_to_append, { i,idx + 1 }, plan.ID, ContainerType::Skyline, score_calc);
                    scores.push_back(score);
                }
                auto itemT = new_item.transpose();
                idx = get_placable_area(itemT, i, plan.SkylineContainers);
                if (idx >= 0) {
                    auto item_to_append = itemT.copy();
                    item_to_append.pos = skyline_rect.start.copy();
                    float score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers, plan.ID);
                    auto score = Score(item_to_append, { i,idx + 1 }, plan.ID, ContainerType::Skyline, score_calc);
                    scores.push_back(score);
                }
            }
        }
    }
    void get_newbin_scores(Item& new_item, vector<Dist3::Score>& scores) {
        vector<Container> fake_container = { Container(Rect(this->material.start, this->material.end)) };
        int idx = get_placable_area(new_item, 0, fake_container);
        if (idx >= 0) {
            auto item_to_append = new_item.copy();
            auto score = Score(item_to_append, { 0,1 }, -1, ContainerType::Skyline, calc_skyline_score(item_to_append, 0, idx, fake_container, -1));
            scores.push_back(score);
        }
        auto itemT = new_item.transpose();
        idx = get_placable_area(itemT, 0, fake_container);
        if (idx >= 0) {
            auto item_to_append = itemT.copy();
            auto score = Score(item_to_append, { 0,1 }, -1, ContainerType::Skyline, calc_skyline_score(item_to_append, 0, idx, fake_container, -1));
            scores.push_back(score);
        }
    }
    void update_skyline_container(Dist3::Plan& plan,Score& best_score, Rect new_rect) {
        auto removed_container_start = plan.SkylineContainers.begin() + best_score.container_range.first;
        auto removed_container_end = plan.SkylineContainers.begin() + best_score.container_range.second;
        vector<Container> removed_containers(removed_container_start, removed_container_end);
        plan.SkylineContainers.erase(removed_container_start, removed_container_end);
        auto& last_c = removed_containers.back();
        optional<Container> maybe_container_right = nullopt;
        optional<Container> maybe_container_top = nullopt;
        maybe_container_top = Container(Rect(new_rect.topLeft(), POS(new_rect.topRight().x, this->material.height())), plan.ID);
        if (new_rect.bottomRight().x != last_c.rect.end.x) {
            maybe_container_right = Container(Rect(POS(new_rect.bottomRight().x, last_c.rect.start.y), POS(last_c.rect.end.x, this->material.height())), plan.ID);
        }
        else {
            if (maybe_container_right.has_value()) {
                maybe_container_right.reset();
            }
        }

        for (auto& sky_c : plan.SkylineContainers) {
            if (not maybe_container_right.has_value() and not maybe_container_top.has_value()) {
                break;
            }
            if (maybe_container_right.has_value()) {
                auto& container_right = maybe_container_right.value();
                auto result = container_right.rect & sky_c.rect;
                if (result == TYPE::LINE) {
                    auto diff = result.end - result.start;
                    if (diff.x == 0) {
                        if (sky_c.rect.bottomLeft() == container_right.rect.bottomRight()) {
                            sky_c.rect.start = container_right.rect.start.copy();
                            maybe_container_right.reset();
                        }
                        else if (sky_c.rect.bottomRight() == container_right.rect.bottomLeft()) {
                            sky_c.rect.end = container_right.rect.end.copy();
                            maybe_container_right.reset();
                        }
                    }
                }

            }
            if (maybe_container_top.has_value()) {
                auto& container_top = maybe_container_top.value();
                auto result = container_top.rect & sky_c.rect;
                if (result == TYPE::LINE) {
                    auto diff = result.end - result.start;
                    if (diff.x == 0) {
                        if (sky_c.rect.bottomLeft() == container_top.rect.bottomRight()) {
                            sky_c.rect.start = container_top.rect.start.copy();
                            maybe_container_top.reset();
                        }
                        else if (sky_c.rect.bottomRight() == container_top.rect.bottomLeft()) {
                            sky_c.rect.end = container_top.rect.end.copy();
                            maybe_container_top.reset();
                        }
                    }
                }

            }

        }
        if (maybe_container_right.has_value()) {
            plan.SkylineContainers.push_back(maybe_container_right.value());
        }
        if (maybe_container_top.has_value()) {
            plan.SkylineContainers.push_back(maybe_container_top.value());
        }
        vector<Container> waste_rect_to_append;
        for (auto i = 1; i < removed_containers.size(); i++) {
            auto& waste_c = removed_containers.at(i);
            if (new_rect.bottomRight().y > waste_c.rect.bottomRight().y) {
                auto c = Container(Rect(waste_c.rect.bottomLeft(), POS(min(waste_c.rect.bottomRight().x, new_rect.bottomRight().x), new_rect.bottomRight().y)), plan.ID);
                waste_rect_to_append.push_back(c);
            }
        }
        for (auto& waste_c : waste_rect_to_append) {
            plan.WasteMap.push_back(waste_c);
        }
    }
    void update_wastemap_container(Dist3::Plan& plan, Score& best_score, Rect new_rect) {
        auto& container = plan.WasteMap.at(best_score.container_range.first);
        // generate the best split plan
        pair<Container, Container> split_1 = {
            Container(Rect(new_rect.topLeft(),container.rect.topRight()),best_score.plan_id),
            Container(Rect(new_rect.bottomRight(),POS(container.rect.bottomRight().x,new_rect.topRight().y),best_score.plan_id))
        };
        pair<Container, Container> split_2 = {
            Container(Rect(new_rect.topLeft(),POS(new_rect.topRight().x,container.rect.topRight().y)),best_score.plan_id),
            Container(Rect(new_rect.bottomRight(),POS(new_rect.topRight().x,container.rect.topRight().y),best_score.plan_id))
        };
        auto split_1_area = max(split_1.first.rect.area(), split_1.second.rect.area());
        auto split_2_area = max(split_2.first.rect.area(), split_2.second.rect.area());

        optional<Container> maybe_newC_top = nullopt;
        optional<Container> maybe_newC_right = nullopt;
        if (split_1_area > split_2_area) {
            maybe_newC_top = split_1.first;
            maybe_newC_right = split_1.second;
        }
        else {
            maybe_newC_top = split_2.first;
            maybe_newC_right = split_2.second;
        }
        if (maybe_newC_top.has_value() && (maybe_newC_top.value() == TYPE::LINE or maybe_newC_top.value() == TYPE::POS)) {
            maybe_newC_top.reset();
        }
        if (maybe_newC_right.has_value() && (maybe_newC_right.value() == TYPE::LINE or maybe_newC_right.value() == TYPE::POS)) {
            maybe_newC_right.reset();
        }
        // maintain the old wastemap, if the new could merge to old
        for (auto& waste_c : plan.WasteMap) {
            if (!maybe_newC_top.has_value() && !maybe_newC_right.has_value()) {
                break;
            }
            if (maybe_newC_right.has_value()) {
                auto& newC_right = maybe_newC_right.value();
                auto result = newC_right.rect & waste_c.rect;
                if (result == TYPE::LINE) {
                    auto diff = result.end - result.start;
                    if (diff.x == 0) {
                        if (waste_c.rect.bottomRight() == newC_right.rect.bottomLeft() and waste_c.rect.topRight() == newC_right.rect.topLeft()) {
                            waste_c.rect.end = newC_right.rect.end.copy();
                            maybe_newC_right.reset();
                        }
                        else if (waste_c.rect.bottomLeft() == newC_right.rect.bottomRight() and waste_c.rect.topLeft() == newC_right.rect.topRight()) {
                            waste_c.rect.start = newC_right.rect.start.copy();
                            maybe_newC_right.reset();
                        }
                    }
                    else if (diff.y == 0) {
                        if (waste_c.rect.topRight() == newC_right.rect.bottomRight() and waste_c.rect.topLeft() == newC_right.rect.bottomLeft()) {
                            waste_c.rect.end = newC_right.rect.end.copy();
                            maybe_newC_right.reset();
                        }
                        else if (waste_c.rect.bottomRight() == newC_right.rect.topRight() and waste_c.rect.bottomLeft() == newC_right.rect.topLeft()) {
                            waste_c.rect.start = newC_right.rect.start.copy();
                            maybe_newC_right.reset();
                        }
                    }
                }
            }
            if (maybe_newC_top.has_value()) {
                auto newC_top = maybe_newC_top.value();
                auto result = newC_top.rect & waste_c.rect;
                if (result == TYPE::LINE) {
                    auto diff = result.end - result.start;
                    if (diff.x == 0) {
                        if (waste_c.rect.bottomRight() == newC_top.rect.bottomLeft() and waste_c.rect.topRight() == newC_top.rect.topLeft()) {
                            waste_c.rect.end = newC_top.rect.end.copy();
                            maybe_newC_top.reset();
                        }
                        else if (waste_c.rect.bottomLeft() == newC_top.rect.bottomRight() and waste_c.rect.topLeft() == newC_top.rect.topRight()) {
                            waste_c.rect.start = newC_top.rect.start.copy();
                            maybe_newC_top.reset();
                        }
                    }
                    else if (diff.y == 0) {
                        if (waste_c.rect.topRight() == newC_top.rect.bottomRight() and waste_c.rect.topLeft() == newC_top.rect.bottomLeft()) {
                            waste_c.rect.end = newC_top.rect.end.copy();
                            maybe_newC_top.reset();
                        }
                        else if (waste_c.rect.bottomRight() == newC_top.rect.topRight() and waste_c.rect.bottomLeft() == newC_top.rect.topLeft()) {
                            waste_c.rect.start = newC_top.rect.start.copy();
                            maybe_newC_top.reset();
                        }
                    }
                }
            }
        }
        if (maybe_newC_right.has_value()) {
            plan.WasteMap.push_back(maybe_newC_right.value());
        }
        if (maybe_newC_top.has_value()) {
            plan.WasteMap.push_back(maybe_newC_top.value());
        }
        erase(plan.WasteMap, container);
    }
    void run() {
        // sorting the items
        sort(this->items.begin(), this->items.end(), [this](Item a,Item b) {
            auto a_rect = a.get_rect();
            auto b_rect = b.get_rect();
            auto a_score = this->scoring_sys.item_sorting(a_rect.width(), a_rect.height());
            auto b_score = this->scoring_sys.item_sorting(b_rect.width(), b_rect.height());
            return a_score < b_score;
            });
        

        solution.clear();// avoid some strange things;
        for (auto item_idx = 0; item_idx < items.size(); item_idx++) { // load item one by one
            auto new_item = items.at(item_idx);
            vector<Dist3::Score> scores; // init a score list;
            get_wastemap_scores(new_item, scores);
            if (scores.size() == 0) {
                get_skyline_scores(new_item, scores);
            }
            if (scores.size() == 0) {
                get_newbin_scores(new_item, scores);
            }
            if (scores.size() == 0) {
                throw runtime_error("no possible item candidates");
            }
            Score best_score = *min_element(scores.begin(), scores.end(), [](const Score& a, const Score& b) {
                if (a.type != b.type) {
                    throw runtime_error("a.type!=b.type");
                }
                return a.score < b.score;
            });


            if (best_score.plan_id == -1) {
                auto plan = Plan(solution.size(), Rect(0, 0, test_material.first, test_material.second));
                auto container_top = Container(Rect(best_score.item.size.topLeft(), POS(best_score.item.size.width(), test_material.second)), plan.ID);
                auto container_right = Container(Rect(best_score.item.size.bottomRight(), this->material.topRight()), plan.ID);
                if (container_top == TYPE::RECT) {
                    plan.SkylineContainers.push_back(container_top);
                }
                if (container_right == TYPE::RECT) {
                    plan.SkylineContainers.push_back(container_right);
                }
                plan.item_sequence.push_back(best_score.item);
                solution.push_back(plan);

                if (is_debug) {
                    PlanPackingLog plan_packing_log;
                    plan_packing_log.push_back(plan.toVector());
                    this->packinglog.push_back(plan_packing_log);
                }
            }
            else {
                auto& plan = solution.at(best_score.plan_id);// load plan
                plan.item_sequence.push_back(best_score.item);// push item
                auto new_rect = best_score.item.get_rect(); // get item actual place position
                if (best_score.type == ContainerType::WasteMap) {
                    update_wastemap_container(plan, best_score, new_rect);
                }
                else {
                    update_skyline_container(plan,best_score,new_rect);
                }
                sort(plan.SkylineContainers.begin(), plan.SkylineContainers.end(), [](Container c1, Container c2) {
                    return c1.rect.start.x < c2.rect.start.x;
                    }
                );
                sort(plan.WasteMap.begin(), plan.WasteMap.end(), [](Container c1, Container c2) {
                    return c1.rect.start.x < c2.rect.start.x;
                    }
                );

                if (is_debug) {
                    auto new_plan = plan;
                    this->packinglog.at(best_score.plan_id).push_back(new_plan.toVector());
                }
                
            }
        
        }
    }
    float calc_skyline_score(Item item, int begin_idx, int end_idx, vector<Container>containers, int plan_id) {

        //float control_term = 1.0f;
        //auto item_rect = item.get_rect();
        //auto last_c = containers.at(end_idx);
        //optional<Rect> maybe_rect_right = nullopt;
        //optional<Rect> maybe_rect_top = nullopt;
        //maybe_rect_top = Rect(item_rect.topLeft(), POS(item_rect.topRight().x, this->material.height()));
        //if (item_rect.bottomRight().x != last_c.rect.end.x) {
        //    maybe_rect_right = Rect(POS(item_rect.bottomRight().x, last_c.rect.start.y), POS(last_c.rect.end.x, this->material.height()));
        //}
        //else {
        //    if (maybe_rect_right.has_value()) {
        //        maybe_rect_right.reset();
        //    }
        //}        
        //float width = item.size.width();
        //float height = item.size.height();
        //float min_y = containers.at(begin_idx).rect.start.y; //at begin idx  skyline container BL corner y
        //float max_gap = containers.size()>1?(*max_element(containers.begin() + 1, containers.end(), [min_y](Container a, Container b) {
        //    return a.rect.bottomLeft().y - min_y < b.rect.bottomLeft().y - min_y;
        //    })).rect.bottomLeft().y:0;
        //float min_gap = containers.size() > 1 ? (*min_element(containers.begin() + 1, containers.end(), [min_y](Container a, Container b) {
        //    return a.rect.bottomLeft().y - min_y < b.rect.bottomLeft().y - min_y;
        //    })).rect.bottomLeft().y : 0;
        //float start_x = containers.at(begin_idx).rect.start.x;
        //float end_x = start_x + width;
        //float waste_area = accumulate(containers.begin(), containers.end(), 0, [min_y](float current_sum, Container b) {
        //    return current_sum + (min_y - b.rect.start.y) * b.rect.width();
        //    }
        //);
        //Rect ideal_waste_rect = accumulate(containers.begin(), containers.end(), Rect(), [](Rect current_sum, Container b) {
        //    return current_sum | b.rect;
        //    }
        //);
        //float container_total_len = accumulate(containers.begin(), containers.end(), 0, [min_y](float current_sum, Container b) {
        //    return current_sum +  b.rect.width();
        //    }
        //);
        /*for (int i = begin_idx; i < end_idx; i++) {
            waste_area += (min_y - containers.at(i + 1).rect.start.y) * containers.at(i + 1).rect.width();
        }*/

        //vector<float> X = { // 14 parameter
        //    float(run_count) / float(items.size()),
        //    solution.size()>0?float(plan_id+1) / float(solution.size()):0,
        //    maybe_rect_top.has_value() ? float(maybe_rect_top.value().area()) / material.area() : 0.0f,
        //    maybe_rect_right.has_value() ? float(maybe_rect_right.value().area() / material.area()) : 0.0f,
        //    maybe_rect_top.has_value() ? maybe_rect_top.value().aspect_ratio() : 0.0f,
        //    maybe_rect_right.has_value() ? maybe_rect_right.value().aspect_ratio() : 0.0f,
        //    item_rect.start.x / material.end.x,
        //    item_rect.start.y / material.end.y,
        //    item_rect.area() / accumulate(containers.begin(),containers.end(),0,[](float current_sum,Container b) {
        //            return current_sum + b.rect.area();
        //        }),
        //    waste_area / ideal_waste_rect.area(),
        //    ideal_waste_rect.aspect_ratio(),
        //    containers.size() > 1 and max_gap > 0 ? (max_gap - min_gap) / max_gap : 0.0f,
        //    item_rect.width() / container_total_len,
        //    item_rect.height() / containers.at(begin_idx).rect.height(),
        //    control_term
        //};
        
        return scoring_sys.skyline_container_pos_scoring(item, begin_idx, end_idx, containers, plan_id);


        //auto p = scoring_sys.get_skyline_pos_scoring_parameters();
        //return inner_product(X.begin(), X.end(), p.begin(), 0);
        /*
        item_id/items.size
        plan_id/solution.size
        top_area/material_area,
        left_area/material_area,
        max(left_width,left_height)/sqrt(left_width**2+left_height**2),
        max(right_width,right_height)/sqrt(right_wdith**2+right_height**2)
        start_y/material_heigth,
        start_x/material_width,
        item_area/total_skyline_container_area,
        total_waste_area/ideal_waste_rect_area,
        ideal_waste_rect_ratio,
        min_waste_gap/max_waste_gap,
        item_width/total_skyline_width,
        item_height/first_skyline_height
        */
        //if (begin_idx == end_idx) {
        //    return { waste_area, min_y };
        //}
        
        //return { waste_area, min_y };


    }
    float calc_wastemap_score(Item item, Container container,int plan_id) {
        /*
        item_id/items.size
        plan_id/solution.size
        item_area/container_area
        item_width/container_width
        item_height/container_height
        container_x/material_width
        container_y/material_height
        item_area/ all_waste_map_area
        control_term
        */

        //auto control_term = 1.0f;
        //auto item_rect = item.get_rect();
        //auto wastemap_containers = solution.at(plan_id).WasteMap;
        //auto all_waste_area = accumulate(wastemap_containers.begin(), wastemap_containers.end(), 0, [](float total_area , Container b) {
        //        return total_area+b.rect.area();
        //    });
        //vector<float> X = { //10 params
        //    float(run_count) / float(items.size()),
        //    solution.size() > 0 ? float(plan_id + 1) / float(solution.size()) : 0,
        //    float(item_rect.area()) / float(container.rect.area()),
        //    1-float(item_rect.area()) / float(container.rect.area()),
        //    item_rect.width() / container.rect.width(),
        //    item_rect.height() / container.rect.height(),
        //    1-container.rect.start.x / material.width(),
        //    1-container.rect.start.y / material.height(),
        //    item_rect.area() / all_waste_area,
        //    control_term,
        //};

        /*auto p = scoring_sys.get_wastemap_pos_scoring_parameters();

        return inner_product(X.begin(), X.end(), p.begin(), 0);*/

        return scoring_sys.wastemap_container_pos_scoring(item, container, plan_id);


    }
    int get_placable_area(Item item, int begin_idx, vector<Container>containers) {

        float height = item.size.height();
        float width = item.size.width();
        float item_start_y = containers.at(begin_idx).rect.start.y;
        float item_start_x = containers.at(begin_idx).rect.start.x;
        float item_end_x = item_start_x + width;
        float item_end_y = item_start_y + height;
        float container_start_y = containers.at(begin_idx).rect.start.y;
        float container_end_y = containers.at(begin_idx).rect.end.y;
        float end_idx = begin_idx;
        if (item_end_x <= containers.at(end_idx).rect.end.x and container_start_y <= item_start_y and item_end_y <= container_end_y) {
            return end_idx;
        }
        else {
            if (end_idx + 1 == containers.size()) {
                return -1;
            }
            for (auto idx = begin_idx + 1; idx < containers.size(); idx++) {
                if (item_end_x <= containers.at(idx).rect.end.x and containers.at(idx).rect.end.y >= item_end_y and item_start_y >= containers.at(idx).rect.start.y) {
                    end_idx = idx;
                    break;
                }
            }
            if (end_idx == begin_idx) {
                return -1;
            }
            for (auto idx = begin_idx; idx < end_idx; idx++) {
                if (not (containers.at(idx).rect.end.y >= item_end_y and item_start_y >= containers.at(idx).rect.start.y)) {
                    return -1;
                }
            }
            return end_idx;
        }
    }
    float get_avg_util_rate() const {
        float total_rate = 0.0;
        for (const auto& plan : this->solution) {
            total_rate += plan.get_util_rate();
        }
        if (this->solution.size() > 0) {
            float ratio = total_rate / this->solution.size();
            return ratio;
        }
        else {
            throw runtime_error("div zero");
        }
    }

    SolutionAsVector solution_as_vector() {
        SolutionAsVector result;
        for (auto plan : this->solution) {
            result.push_back(plan.toVector());
        }
        return result;
    }


};






//-------------------------------MaxRect algo-------------------------------------------------------------
class MaxRect :public Algo {
public:
    MaxRect(vector<float> flat_items, pair<float, float> material = { 2440,1220 }, string task_id = "", bool is_debug = false) :Algo(flat_items, material, task_id, is_debug) {

    }
    void run() {
        solution.clear();

        while (items.size() > 0)
        {
            auto scores_li = find_possible_item_candidates();
            auto best_score = *std::min_element(scores_li.begin(), scores_li.end(), [](const ItemScore& p1, const ItemScore& p2) {
                return p1.getScore() < p2.getScore();
                });

            if (best_score.plan_id == -1) {
                auto plan = ProtoPlan(solution.size(), material);
                plan.item_sequence.push_back(best_score.item);
                auto item_rect = best_score.item.get_rect();
                auto container_top = Container(Rect(item_rect.topLeft(), material.topRight()), plan.ID);
                auto container_btm = Container(Rect(item_rect.bottomRight(), material.topRight()), plan.ID);
                plan.remain_containers.push_back(container_top);
                plan.remain_containers.push_back(container_btm);
                solution.push_back(plan);

                if (is_debug) {
                    PlanPackingLog planlog;
                    planlog.push_back(plan.toVector());
                    this->packinglog.push_back(planlog);
                }
            }
            else {
                auto& plan = solution.at(best_score.plan_id);
                plan.item_sequence.push_back(best_score.item);
                auto new_item = best_score.item;
                auto container = best_score.container;
                auto new_rect = new_item.get_rect();
                update_remain_containers(new_item, plan);
                if (is_debug) {
                    this->packinglog.at(best_score.plan_id).push_back(plan.toVector());
                }
            }

            erase(items, best_score.item);
        }
    }
    vector<ItemScore> find_possible_item_candidates() {
        vector<ItemScore> scores_li;

        for (auto i = 0; i < items.size(); i++)
        {
            Item& item = items.at(i);
            // from existing containers
            for (ProtoPlan& plan : solution)
            {

                for (auto& container : plan.remain_containers)
                {
                    Item item_prepare = item.copy();
                    checkAndScoreItem(item_prepare, container, scores_li);
                    Item item_prepareT = item.copy().transpose();
                    checkAndScoreItem(item_prepareT, container, scores_li);
                }
            }
            Container fake_container = Container(material.copy());
            Item item_prepare = item.copy();
            checkAndScoreItem(item_prepare, fake_container, scores_li);
            Item item_prepareT = item.transpose();
            checkAndScoreItem(item_prepareT, fake_container, scores_li);

        }

        /*if (scores_li.size() == 0){
            for (auto i = 0; i < items.size(); i++){
                Item& item = items[i];
                Container fake_container = Container(material.copy());
                Item item_prepare = item.copy();
                checkAndScoreItem(item_prepare, fake_container, scores_li);
                Item item_prepareT = item.transpose();
                checkAndScoreItem(item_prepareT, fake_container, scores_li);
            }
        }*/

        if (scores_li.size() == 0) {
            throw runtime_error("no possible item candidates");
        }
        return scores_li;
    }
    void checkAndScoreItem(Item item, Container& container, std::vector<ItemScore>& score_list) {
        auto corner_start = container.rect.start;
        if (container.contains(item.size + corner_start)) {
            auto item_to_append = item.copy();
            item_to_append.pos = corner_start.copy();
            float score = min(container.rect.width() - item.size.width(), container.rect.height() - item.size.height());
            score_list.push_back(ItemScore(item_to_append, container, score));
        }
    }
    void update_remain_containers(Item& item, ProtoPlan& plan) {
        vector<Container>container_to_remove;
        vector<optional<Container>>container_to_append;
        //split intersect rect
        /*for(auto& free_c : plan.remain_containers){
            auto result = free_c.rect & item.get_rect();
            if(result == TYPE::RECT){
                container_to_remove.push_back(free_c);
                if(result.topRight().y < free_c.rect.topRight().y){
                    auto top_c = Container(Rect(POS(free_c.rect.topLeft().x,result.topLeft().y),free_c.rect.topRight()),plan.ID);
                    if(top_c.rect == TYPE::RECT){
                        container_to_append.push_back(top_c);
                    }
                }
                if(result.bottomRight().y > free_c.rect.bottomRight().y){
                    auto btm_c = Container(Rect(free_c.rect.bottomLeft(),POS(free_c.rect.bottomRight().x,result.bottomRight().y)),plan.ID);
                    if(btm_c.rect == TYPE::RECT){
                        container_to_append.push_back(btm_c);
                    }
                }
                if(result.topRight().x < free_c.rect.topRight().x){
                    auto left_c = Container(Rect(POS(result.bottomRight().x,free_c.rect.bottomLeft().y),free_c.rect.topRight()),plan.ID);
                    if(left_c.rect == TYPE::RECT){
                        container_to_append.push_back(left_c);
                    }
                }
                if(result.topLeft().x > free_c.rect.topLeft().x){
                    auto right_c = Container(Rect(free_c.rect.bottomLeft(),POS(result.topLeft().x,free_c.rect.topRight().y)),plan.ID);
                    if(right_c.rect == TYPE::RECT){
                        container_to_append.push_back(right_c);
                    }
                }
            }
        }

        for(const auto c : container_to_remove){
            //erase(plan.remain_containers, c);
            plan.remain_containers.erase(std::remove(plan.remain_containers.begin(), plan.remain_containers.end(), c), plan.remain_containers.end());
            if (is_debug) {
                auto plancopy = plan;
                this->packinglog.at(plan.ID).push_back(plan.toVector(false));
            }
        }
        // merge rect
        for(auto& free_c: plan.remain_containers){
            for(auto i=0;i<container_to_append.size();i++){
                if(container_to_append[i].has_value()){
                    auto result = container_to_append[i].value().rect;
                    if((result & free_c.rect) == result){
                        container_to_append[i].reset();
                    }
                    else if (result == TYPE::LINE) {
                        auto diff = result.end - result.start;
                        if (diff.x == 0) {
                            if (result.start == free_c.rect.bottomRight()) {
                                free_c.rect.end = container_to_append[i].value().rect.end.copy();
                            }
                            else if (result.start == free_c.rect.bottomLeft()) {
                                free_c.rect.start = container_to_append[i].value().rect.start.copy();
                            }
                        }
                        else if (diff.y == 0) {
                            if (result.end == free_c.rect.topRight()) {
                                free_c.rect.end = container_to_append[i].value().rect.end.copy();
                            }
                            else if (result.end == free_c.rect.bottomRight()) {
                                free_c.rect.start = container_to_append[i].value().rect.start.copy();
                            }
                        }
                        container_to_append[i].reset();
                    }
                }
            }
        }
        for(auto& container : container_to_append){
            if (container.has_value() && (find(plan.remain_containers.begin(), plan.remain_containers.end(), container.value()) == plan.remain_containers.end())) {
                plan.remain_containers.push_back(container.value());
                if (is_debug) {
                    auto plancopy = plan;
                    this->packinglog.at(plan.ID).push_back(plan.toVector(false));
                }
            }
        }
        */

        auto item_rect = item.get_rect();
        // split 
        for (auto& container : plan.remain_containers) {
            /*if ((container.rect & item_rect) != TYPE::RECT) {
                continue;
            }*/
            RectDivided result_rects = container.rect / item_rect;
            bool remove = false;
            for (auto& r : result_rects) {
                if (r.has_value()) {
                    remove = true;
                    container_to_append.push_back(Container(r.value(), plan.ID));
                }
            }
            if (remove) {
                container_to_remove.push_back(container);
            }

        }
        // remove contained rect from new splited rect
        for (auto& containerA : container_to_append) {
            if (!containerA.has_value()) {
                continue;
            }
            for (auto& containerB : container_to_append) {
                if (!containerB.has_value()) {
                    continue;
                }
                if (containerA.value() == containerB.value()) {
                    continue;
                }
                if (containerA.value().contains(containerB.value())) {
                    containerB.reset();
                }
                if (containerB.has_value() and containerB.value().contains(containerA.value())) {
                    containerA.reset();
                    break;
                }
            }
        }
        // remove splited rect from old container
        for (auto& container : container_to_remove) {
            std::erase(plan.remain_containers, container);
            if (is_debug) {
                auto plancopy = plan;
                this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
            }
        }

        // check if new rect is contained
        for (auto& new_container : container_to_append) {
            if (!new_container.has_value()) {
                continue;
            }
            for (auto& old_container : plan.remain_containers) {
                if (old_container.rect.contains(new_container.value().rect)) {
                    new_container.reset();
                    break;
                }
            }
            if (new_container.has_value()) {
                plan.remain_containers.push_back(new_container.value());
                if (is_debug) {
                    auto plancopy = plan;
                    this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
                }
            }
        }


    }
};


class MAXRECTS_BSSF_BBF_DESCSS :public Algo {
public:
    MAXRECTS_BSSF_BBF_DESCSS(vector<float> flat_items, pair<float, float> material = { 2440,1220 }, string task_id = "", bool is_debug = false) :Algo(flat_items, material, task_id, is_debug) {

    }
    void run() {
        solution.clear();
        sort(this->items.begin(), this->items.end(), [this](Item a, Item b) {
                return a.size.width() > b.size.width();
            });

        for (auto i = 0; i < items.size(); i++) {
            auto new_item = items.at(i);
            auto scores_li = find_possible_item_candidates(new_item);
            auto best_score = *std::min_element(scores_li.begin(), scores_li.end(), [](const ItemScore& p1, const ItemScore& p2) {
                return p1.getScore() < p2.getScore();
                });
            if (best_score.plan_id == -1) {
                auto plan = ProtoPlan(solution.size(), material);
                plan.item_sequence.push_back(best_score.item);
                auto item_rect = best_score.item.get_rect();
                auto container_top = Container(Rect(item_rect.topLeft(), material.topRight()), plan.ID);
                auto container_btm = Container(Rect(item_rect.bottomRight(), material.topRight()), plan.ID);
                plan.remain_containers.push_back(container_top);
                plan.remain_containers.push_back(container_btm);
                solution.push_back(plan);

                if (is_debug) {
                    PlanPackingLog planlog;
                    planlog.push_back(plan.toVector());
                    this->packinglog.push_back(planlog);
                }
            }
            else {
                auto& plan = solution.at(best_score.plan_id);
                plan.item_sequence.push_back(best_score.item);
                auto new_item = best_score.item;
                auto container = best_score.container;
                auto new_rect = new_item.get_rect();
                update_remain_containers(new_item, plan);
                if (is_debug) {
                    this->packinglog.at(best_score.plan_id).push_back(plan.toVector());
                }
            }

        }

    }
    vector<ItemScore> find_possible_item_candidates(Item item) {
        vector<ItemScore> scores_li;

       
        // from existing containers
        for (ProtoPlan& plan : solution)
        {

            for (auto& container : plan.remain_containers)
            {
                Item item_prepare = item.copy();
                checkAndScoreItem(item_prepare, container, scores_li);
                Item item_prepareT = item.copy().transpose();
                checkAndScoreItem(item_prepareT, container, scores_li);
            }
        }
        Container fake_container = Container(material.copy());
        Item item_prepare = item.copy();
        checkAndScoreItem(item_prepare, fake_container, scores_li);
        Item item_prepareT = item.transpose();
        checkAndScoreItem(item_prepareT, fake_container, scores_li);

        if (scores_li.size() == 0) {
            throw runtime_error("no possible item candidates");
        }
        return scores_li;
    }
    void checkAndScoreItem(Item item, Container& container, std::vector<ItemScore>& score_list) {
        auto corner_start = container.rect.start;
        if (container.contains(item.size + corner_start)) {
            auto item_to_append = item.copy();
            item_to_append.pos = corner_start.copy();
            float score = min(container.rect.width() - item.size.width(), container.rect.height() - item.size.height()); //BSSF
            score_list.push_back(ItemScore(item_to_append, container, score));
        }
    }
    void update_remain_containers(Item& item, ProtoPlan& plan) {
        vector<Container>container_to_remove;
        vector<optional<Container>>container_to_append;
        
        auto item_rect = item.get_rect();
        // split 
        for (auto& container : plan.remain_containers) {
            /*if ((container.rect & item_rect) != TYPE::RECT) {
                continue;
            }*/
            RectDivided result_rects = container.rect / item_rect;
            bool remove = false;
            for (auto& r : result_rects) {
                if (r.has_value()) {
                    remove = true;
                    container_to_append.push_back(Container(r.value(), plan.ID));
                }
            }
            if (remove) {
                container_to_remove.push_back(container);
            }

        }
        // remove contained rect from new splited rect
        for (auto& containerA : container_to_append) {
            if (!containerA.has_value()) {
                continue;
            }
            for (auto& containerB : container_to_append) {
                if (!containerB.has_value()) {
                    continue;
                }
                if (containerA.value() == containerB.value()) {
                    continue;
                }
                if (containerA.value().contains(containerB.value())) {
                    containerB.reset();
                }
                if (containerB.has_value() and containerB.value().contains(containerA.value())) {
                    containerA.reset();
                    break;
                }
            }
        }
        // remove splited rect from old container
        for (auto& container : container_to_remove) {
            std::erase(plan.remain_containers, container);
            if (is_debug) {
                auto plancopy = plan;
                this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
            }
        }

        // check if new rect is contained
        for (auto& new_container : container_to_append) {
            if (!new_container.has_value()) {
                continue;
            }
            for (auto& old_container : plan.remain_containers) {
                if (old_container.rect.contains(new_container.value().rect)) {
                    new_container.reset();
                    break;
                }
            }
            if (new_container.has_value()) {
                plan.remain_containers.push_back(new_container.value());
                if (is_debug) {
                    auto plancopy = plan;
                    this->packinglog.at(plan.ID).push_back(plancopy.toVector(false));
                }
            }
        }


    }
};



class Skyline :public Algo {

public:
    class Plan :public ProtoPlan {
    public:

        vector<Container> SkylineContainers;
        vector<Container> WasteMap;
        Plan(int ID, Rect material, vector<Item> item_sequence = vector<Item>(), vector<Container> remain_containers = vector<Container>()) :
            ProtoPlan(ID, material) {};
        vector<Container> get_remain_containers(bool throw_error = true)const override {
            vector<Container> result;
            for (auto c : SkylineContainers) {
                result.push_back(c);
            }
            for (auto c : WasteMap) {
                result.push_back(c);
            }
            return result;
        }

    };
    enum class ContainerType {
        WasteMap,
        Skyline
    };
    class Score {
    public:
        Item item;
        pair<int, int> container_range;
        float score;
        pair<float, float>scores;
        int plan_id;
        Skyline::ContainerType type;
        Score(Item item, pair<int, int> container_range, int plan_id, Skyline::ContainerType type, float score = 0, pair<float, float>scores = { 0,0 }) :item(item), container_range(container_range), score(score), scores(scores), plan_id(plan_id), type(type) {};
    };

    vector<Skyline::Plan>solution;
    Skyline(vector<float> flat_items, pair<float, float> material = { 2440,1220 }, string task_id = "", bool is_debug = false) :Algo(flat_items, material, task_id, is_debug) {}
    void run() {
        // sorting the items
        sort(this->items.begin(), this->items.end(), [](const Item& a, const Item& b) {
            float min_side_a = min(a.size.width(), a.size.height());
            float min_side_b = min(b.size.width(), b.size.height());
            return min_side_a > min_side_b;
            });
        if (this->is_debug) {
            cout << "sorted_item:\n";
            for (auto item : items) {
                cout << item.get_rect().to_string() << endl;
            }
        }

        solution.clear();// avoid some strange things;
        for (auto& new_item : items) { // load item one by one
            if (this->is_debug) { cout << "solution size=" << solution.size() << endl; }
            vector<Skyline::Score> scores; // init a score list;
            for (auto& plan : solution) {
                for (auto i = 0; i < plan.WasteMap.size(); i++) {
                    Rect waste_rect = plan.WasteMap.at(i).rect;
                    if (waste_rect.contains(new_item.size + waste_rect.start)) {
                        auto item_to_append = new_item.copy();
                        item_to_append.pos = waste_rect.start.copy();
                        auto score = Score(item_to_append, { i,i + 1 }, plan.ID, ContainerType::WasteMap, calc_wastemap_score(item_to_append, plan.WasteMap.at(i)));
                        scores.push_back(score);
                    }
                    auto itemT = new_item.transpose();
                    if (waste_rect.contains(itemT.size + waste_rect.start)) {
                        auto item_to_append = itemT.copy();
                        item_to_append.pos = waste_rect.start.copy();
                        auto score = Score(item_to_append, { i,i + 1 }, plan.ID, ContainerType::WasteMap, calc_wastemap_score(item_to_append, plan.WasteMap.at(i)));
                        scores.push_back(score);
                    }
                }
            }
            if (scores.size() == 0) {
                for (auto& plan : solution) {
                    for (auto i = 0; i < plan.SkylineContainers.size(); i++) {
                        Rect skyline_rect = plan.SkylineContainers.at(i).rect;
                        int idx = get_placable_area(new_item, i, plan.SkylineContainers);
                        if (idx >= 0) {
                            auto item_to_append = new_item.copy();
                            item_to_append.pos = skyline_rect.start.copy();
                            pair<float, float> score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers);
                            auto score = Score(item_to_append, { i,idx + 1 }, plan.ID, ContainerType::Skyline, 0, score_calc);
                            scores.push_back(score);
                        }
                        auto itemT = new_item.transpose();
                        idx = get_placable_area(itemT, i, plan.SkylineContainers);
                        if (idx >= 0) {
                            auto item_to_append = itemT.copy();
                            item_to_append.pos = skyline_rect.start.copy();
                            pair<float, float> score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers);
                            auto score = Score(item_to_append, { i,idx + 1 }, plan.ID, ContainerType::Skyline, 0, score_calc);
                            scores.push_back(score);
                        }
                    }
                }
            }
            if (scores.size() == 0) {
                vector<Container> fake_container = { Container(Rect(this->material.start, this->material.end)) };
                int idx = get_placable_area(new_item, 0, fake_container);
                if (idx >= 0) {
                    auto item_to_append = new_item.copy();
                    auto score = Score(item_to_append, { 0,1 }, -1, ContainerType::Skyline, 0, calc_skyline_score(item_to_append, 0, idx, fake_container));
                    scores.push_back(score);
                }
                auto itemT = new_item.transpose();
                idx = get_placable_area(itemT, 0, fake_container);
                if (idx >= 0) {
                    auto item_to_append = itemT.copy();
                    auto score = Score(item_to_append, { 0,1 }, -1, ContainerType::Skyline, 0, calc_skyline_score(item_to_append, 0, idx, fake_container));
                    scores.push_back(score);
                }
            }
            if (scores.size() == 0) {
                throw runtime_error("no possible item candidates");
            }
            Score best_score = *min_element(scores.begin(), scores.end(), [](const Score& a, const Score& b) {
                if (a.type != b.type) {
                    throw runtime_error("a.type!=b.type");
                }
                if (a.plan_id == b.plan_id) {
                    if (a.type == ContainerType::WasteMap) {
                        return a.score < b.score;
                    }
                    else {
                        if (a.scores.first == b.scores.first) {
                            return a.scores.second < b.scores.second;
                        }
                        else {
                            return a.scores.first < b.scores.first;
                        }
                    }
                }
                else {
                    return a.plan_id < b.plan_id;
                }
                });



            if (best_score.plan_id == -1) {
                auto plan = Plan(solution.size(), Rect(0, 0, test_material.first, test_material.second));
                auto container_top = Container(Rect(best_score.item.size.topLeft(), POS(best_score.item.size.width(), test_material.second)), plan.ID);
                auto container_right = Container(Rect(best_score.item.size.bottomRight(), this->material.topRight()), plan.ID);
                if (container_top == TYPE::RECT) {
                    plan.SkylineContainers.push_back(container_top);
                }
                if (container_right == TYPE::RECT) {
                    plan.SkylineContainers.push_back(container_right);
                }
                plan.item_sequence.push_back(best_score.item);
                solution.push_back(plan);

                PlanPackingLog plan_packing_log;
                plan_packing_log.push_back(plan.toVector());
                this->packinglog.push_back(plan_packing_log);
            }
            else {
                auto& plan = solution.at(best_score.plan_id);// load plan
                plan.item_sequence.push_back(best_score.item);// push item

                auto new_rect = best_score.item.get_rect(); // get item actual place position
                if (best_score.type == ContainerType::WasteMap) {
                    auto& container = plan.WasteMap.at(best_score.container_range.first);
                    // generate the best split plan
                    pair<Container, Container> split_1 = {
                        Container(Rect(new_rect.topLeft(),container.rect.topRight()),best_score.plan_id),
                        Container(Rect(new_rect.bottomRight(),POS(container.rect.bottomRight().x,new_rect.topRight().y),best_score.plan_id))
                    };
                    pair<Container, Container> split_2 = {
                        Container(Rect(new_rect.topLeft(),POS(new_rect.topRight().x,container.rect.topRight().y)),best_score.plan_id),
                        Container(Rect(new_rect.bottomRight(),POS(new_rect.topRight().x,container.rect.topRight().y),best_score.plan_id))
                    };
                    auto split_1_area = max(split_1.first.rect.area(), split_1.second.rect.area());
                    auto split_2_area = max(split_2.first.rect.area(), split_2.second.rect.area());

                    optional<Container> maybe_newC_top = nullopt;
                    optional<Container> maybe_newC_right = nullopt;
                    if (split_1_area > split_2_area) {
                        maybe_newC_top = split_1.first;
                        maybe_newC_right = split_1.second;
                    }
                    else {
                        maybe_newC_top = split_2.first;
                        maybe_newC_right = split_2.second;
                    }
                    if (maybe_newC_top.has_value() && (maybe_newC_top.value() == TYPE::LINE or maybe_newC_top.value() == TYPE::POS)) {
                        maybe_newC_top.reset();
                    }
                    if (maybe_newC_right.has_value() && (maybe_newC_right.value() == TYPE::LINE or maybe_newC_right.value() == TYPE::POS)) {
                        maybe_newC_right.reset();
                    }
                    // maintain the old wastemap, if the new could merge to old
                    for (auto& waste_c : plan.WasteMap) {
                        if (!maybe_newC_top.has_value() && !maybe_newC_right.has_value()) {
                            break;
                        }
                        if (maybe_newC_right.has_value()) {
                            auto& newC_right = maybe_newC_right.value();
                            auto result = newC_right.rect & waste_c.rect;
                            if (result == TYPE::LINE) {
                                auto diff = result.end - result.start;
                                if (diff.x == 0) {
                                    if (waste_c.rect.bottomRight() == newC_right.rect.bottomLeft() and waste_c.rect.topRight() == newC_right.rect.topLeft()) {
                                        waste_c.rect.end = newC_right.rect.end.copy();
                                        maybe_newC_right.reset();
                                    }
                                    else if (waste_c.rect.bottomLeft() == newC_right.rect.bottomRight() and waste_c.rect.topLeft() == newC_right.rect.topRight()) {
                                        waste_c.rect.start = newC_right.rect.start.copy();
                                        maybe_newC_right.reset();
                                    }
                                }
                                else if (diff.y == 0) {
                                    if (waste_c.rect.topRight() == newC_right.rect.bottomRight() and waste_c.rect.topLeft() == newC_right.rect.bottomLeft()) {
                                        waste_c.rect.end = newC_right.rect.end.copy();
                                        maybe_newC_right.reset();
                                    }
                                    else if (waste_c.rect.bottomRight() == newC_right.rect.topRight() and waste_c.rect.bottomLeft() == newC_right.rect.topLeft()) {
                                        waste_c.rect.start = newC_right.rect.start.copy();
                                        maybe_newC_right.reset();
                                    }
                                }
                            }
                        }
                        if (maybe_newC_top.has_value()) {
                            auto newC_top = maybe_newC_top.value();
                            auto result = newC_top.rect & waste_c.rect;
                            if (result == TYPE::LINE) {
                                auto diff = result.end - result.start;
                                if (diff.x == 0) {
                                    if (waste_c.rect.bottomRight() == newC_top.rect.bottomLeft() and waste_c.rect.topRight() == newC_top.rect.topLeft()) {
                                        waste_c.rect.end = newC_top.rect.end.copy();
                                        maybe_newC_top.reset();
                                    }
                                    else if (waste_c.rect.bottomLeft() == newC_top.rect.bottomRight() and waste_c.rect.topLeft() == newC_top.rect.topRight()) {
                                        waste_c.rect.start = newC_top.rect.start.copy();
                                        maybe_newC_top.reset();
                                    }
                                }
                                else if (diff.y == 0) {
                                    if (waste_c.rect.topRight() == newC_top.rect.bottomRight() and waste_c.rect.topLeft() == newC_top.rect.bottomLeft()) {
                                        waste_c.rect.end = newC_top.rect.end.copy();
                                        maybe_newC_top.reset();
                                    }
                                    else if (waste_c.rect.bottomRight() == newC_top.rect.topRight() and waste_c.rect.bottomLeft() == newC_top.rect.topLeft()) {
                                        waste_c.rect.start = newC_top.rect.start.copy();
                                        maybe_newC_top.reset();
                                    }
                                }
                            }
                        }
                    }
                    if (maybe_newC_right.has_value()) {
                        plan.WasteMap.push_back(maybe_newC_right.value());
                    }
                    if (maybe_newC_top.has_value()) {
                        plan.WasteMap.push_back(maybe_newC_top.value());
                    }
                    erase(plan.WasteMap, container);

                }
                else {
                    auto removed_container_start = plan.SkylineContainers.begin() + best_score.container_range.first;
                    auto removed_container_end = plan.SkylineContainers.begin() + best_score.container_range.second;
                    vector<Container> removed_containers(removed_container_start, removed_container_end);
                    plan.SkylineContainers.erase(removed_container_start, removed_container_end);
                    auto& last_c = removed_containers.back();
                    optional<Container> maybe_container_right = nullopt;
                    optional<Container> maybe_container_top = nullopt;
                    maybe_container_top = Container(Rect(new_rect.topLeft(), POS(new_rect.topRight().x, this->material.height())), plan.ID);
                    if (new_rect.bottomRight().x != last_c.rect.end.x) {
                        maybe_container_right = Container(Rect(POS(new_rect.bottomRight().x, last_c.rect.start.y), POS(last_c.rect.end.x, this->material.height())),plan.ID);
                    }
                    else {
                        if (maybe_container_right.has_value()) {
                            maybe_container_right.reset();
                        }
                    }

                    for (auto& sky_c : plan.SkylineContainers) {
                        if (not maybe_container_right.has_value() and not maybe_container_top.has_value()) {
                            break;
                        }
                        if (maybe_container_right.has_value()) {
                            auto& container_right = maybe_container_right.value();
                            auto result = container_right.rect & sky_c.rect;
                            if (result == TYPE::LINE) {
                                auto diff = result.end - result.start;
                                if (diff.x == 0) {
                                    if (sky_c.rect.bottomLeft() == container_right.rect.bottomRight()) {
                                        sky_c.rect.start = container_right.rect.start.copy();
                                        maybe_container_right.reset();
                                    }
                                    else if (sky_c.rect.bottomRight() == container_right.rect.bottomLeft()) {
                                        sky_c.rect.end = container_right.rect.end.copy();
                                        maybe_container_right.reset();
                                    }
                                }
                            }

                        }
                        if (maybe_container_top.has_value()) {
                            auto& container_top = maybe_container_top.value();
                            auto result = container_top.rect & sky_c.rect;
                            if (result == TYPE::LINE) {
                                auto diff = result.end - result.start;
                                if (diff.x == 0) {
                                    if (sky_c.rect.bottomLeft() == container_top.rect.bottomRight()) {
                                        sky_c.rect.start = container_top.rect.start.copy();
                                        maybe_container_top.reset();
                                    }
                                    else if (sky_c.rect.bottomRight() == container_top.rect.bottomLeft()) {
                                        sky_c.rect.end = container_top.rect.end.copy();
                                        maybe_container_top.reset();
                                    }
                                }
                            }

                        }

                    }
                    if (maybe_container_right.has_value()) {
                        plan.SkylineContainers.push_back(maybe_container_right.value());
                    }
                    if (maybe_container_top.has_value()) {
                        plan.SkylineContainers.push_back(maybe_container_top.value());
                    }
                    vector<Container> waste_rect_to_append;
                    for (auto i = 1; i < removed_containers.size(); i++) {
                        auto& waste_c = removed_containers.at(i);
                        if (new_rect.bottomRight().y > waste_c.rect.bottomRight().y) {
                            auto c = Container(Rect(waste_c.rect.bottomLeft(), POS(min(waste_c.rect.bottomRight().x, new_rect.bottomRight().x), new_rect.bottomRight().y)), plan.ID);
                            waste_rect_to_append.push_back(c);
                        }
                    }
                    for (auto& waste_c : waste_rect_to_append) {
                        plan.WasteMap.push_back(waste_c);
                    }
                }
                sort(plan.SkylineContainers.begin(), plan.SkylineContainers.end(), [](Container c1, Container c2) {
                    return c1.rect.start.x < c2.rect.start.x;
                    }
                );
                sort(plan.WasteMap.begin(), plan.WasteMap.end(), [](Container c1, Container c2) {
                    return c1.rect.start.x < c2.rect.start.x;
                    }
                );

                auto new_plan = plan;
                this->packinglog.at(best_score.plan_id).push_back(new_plan.toVector());
            }
        }
    
    
    
    }
    pair<float, float> calc_skyline_score(Item item, int begin_idx, int end_idx, vector<Container>containers) {
        if (this->is_debug)
        {
            cout << "calc_skyline_score.begin_idx=" << begin_idx << ",end_idx=" << end_idx << ",containers.size" << containers.size() << endl;
        }

        float waste_area = 0;
        float width = item.size.width();
        float height = item.size.height();
        float min_y = containers.at(begin_idx).rect.start.y;
        float start_x = containers.at(begin_idx).rect.start.x;
        float end_x = start_x + width;
        if (begin_idx == end_idx) {
            return { waste_area, min_y };
        }
        for (int i = begin_idx; i < end_idx; i++) {
            waste_area += (min_y - containers.at(i + 1).rect.start.y) * containers.at(i + 1).rect.width();
        }
        return { waste_area, min_y };

    }
    float calc_wastemap_score(Item item, Container container) {
        if (item.size.height() > item.size.width()) {
            return container.rect.width() - item.size.width();
        }
        else {
            return container.rect.height() - item.size.height();
        }
    }
    int get_placable_area(Item item, int begin_idx, vector<Container>containers) {
        if (this->is_debug)
        {
            cout << "get_placable_area.begin_idx=" << begin_idx << ",containers.size" << containers.size() << endl;
        }
        float height = item.size.height();
        float width = item.size.width();
        float item_start_y = containers.at(begin_idx).rect.start.y;
        float item_start_x = containers.at(begin_idx).rect.start.x;
        float item_end_x = item_start_x + width;
        float item_end_y = item_start_y + height;
        float container_start_y = containers.at(begin_idx).rect.start.y;
        float container_end_y = containers.at(begin_idx).rect.end.y;
        float end_idx = begin_idx;
        if (item_end_x <= containers.at(end_idx).rect.end.x and container_start_y <= item_start_y and item_end_y <= container_end_y) {
            return end_idx;
        }
        else {
            if (end_idx + 1 == containers.size()) {
                return -1;
            }
            for (auto idx = begin_idx + 1; idx < containers.size(); idx++) {
                if (item_end_x <= containers.at(idx).rect.end.x and containers.at(idx).rect.end.y >= item_end_y and item_start_y >= containers.at(idx).rect.start.y) {
                    end_idx = idx;
                    break;
                }
            }
            if (end_idx == begin_idx) {
                return -1;
            }
            for (auto idx = begin_idx; idx < end_idx; idx++) {
                if (not (containers.at(idx).rect.end.y >= item_end_y and item_start_y >= containers.at(idx).rect.start.y)) {
                    return -1;
                }
            }
            return end_idx;
        }
    }
    float get_avg_util_rate() const {
        double total_rate = 0.0;
        for (const auto& plan : this->solution) {
            total_rate += plan.get_util_rate();
        }
        if (this->solution.size() > 0) {
            double ratio = total_rate / this->solution.size();
            return ratio;
        }
        else {
            throw runtime_error("div zero");
        }
    }

    SolutionAsVector solution_as_vector() {
        SolutionAsVector result;
        for (auto plan : this->solution) {
            result.push_back(plan.toVector());
        }
        return result;
    }

};

struct AlgoName {
    static constexpr const char* Dist = "Dist";
    static constexpr const char* Dist_MaxRect = "Dist_MaxRect";
    static constexpr const char* Dist_Skyline = "Dist_Skyline";
    static constexpr const char* Dist_Shelf = "Dist_Shelf";
    static constexpr const char* MaxRect = "MaxRect";
    static constexpr const char* Skyline = "Skyline";
};


std::unique_ptr<Algo> inner_single_run(vector<float> items, pair<float, float>material, std::optional<vector<float>> parameter_input_array = std::nullopt, string algo_type = AlgoName::Dist, bool is_debug = false) {
    if (is_debug) {
        cout << "items size = " << items.size()<<" , algo_name="<<algo_type << endl;
    }
    if (algo_type ==  AlgoName::Dist) {
        if (parameter_input_array.has_value()) {
            vector<float> parameter_items = parameter_input_array.value();
            auto d = std::make_unique<Dist>(items, material);
            d->scoring_sys.parameters = parameter_items;
            d->run();
            return d;
        }
        else {
            auto d = std::make_unique<Dist>(items, material);
            d->run();
            return d;
        }
    }
    else if (algo_type == AlgoName::MaxRect) {
        auto d = std::make_unique<MAXRECTS_BSSF_BBF_DESCSS>(items, material);
        d->run();
        return d;
    }
    else if (algo_type == AlgoName::Skyline) {
        auto d = std::make_unique<Skyline>(items, material);
        d->run();
        return d;
    }
    else if (algo_type == AlgoName::Dist_MaxRect) {
        if (parameter_input_array.has_value()) {
            vector<float> parameter_items = parameter_input_array.value();
            auto d = std::make_unique<Dist2>(items, material, "", is_debug);
            d->scoring_sys.parameters = parameter_items;
            d->run();
            return d;
        }
        else {
            auto d = std::make_unique<Dist2>(items, material, "", is_debug);
            d->run();
            return d;
        }
    }
    else if (algo_type == AlgoName::Dist_Skyline) {
        if (parameter_input_array.has_value()) {
            vector<float> parameter_items = parameter_input_array.value();
            auto d = std::make_unique<Dist3>(items, material, "", is_debug);
            d->scoring_sys.parameters = parameter_items;
            d->run();
            return d;
        }
        else {
            auto d = std::make_unique<Dist3>(items, material, "", is_debug);
            d->run();
            return d;
        }
    }
    throw runtime_error("Algorithm not found for given algo_type.");
}

int get_algo_parameters_length(string algo_type) {
    if (algo_type == AlgoName::Dist) {
        return Dist::ScoringSys::total_param_count;
    }
        
    else if (algo_type == AlgoName::Dist_MaxRect) {
        return Dist2::ScoringSys::ParameterCount::total;
    }
    else if (algo_type == AlgoName::Dist_Skyline) {
        return Dist3::ScoringSys::ParameterCount::total;
    }
    else {
        
        throw runtime_error("can find algo");
    }
}


void test_dist_skyline() {
    vector<float> p = {
-28.87387150923436, -28.618529007833864, 26.980629974625657, -20.242333126873046, -17.267807422076203, -1.5575831483146878, 11.563719590419268, 19.65316489192614, 27.48682202245689, -23.152815136458504, 7.0367714165386275, -31.0, -6.375059883757615, 20.186779203860247, -14.635306910358253, -1.882856296641858, -21.562415817508672, -0.03759380610212659, 28.698550292677588, 13.58471139424043, 9.301246765693946, -31.0, -18.587302872532703, -0.7723136578646965, 24.411996550705233, -9.603301322576229, 14.775923677247341, -24.081146356445526, 31.0, 1.872020721505102, -8.947888456894628
    };

    //test_rect();
    std::default_random_engine generator;
    for (auto x = 0; x < 4; x++) {
        cout << "iter=" << x << endl;
        for (auto j = 0; j < 5; j++) {
            vector<float>input_data;
            for (auto i = 0; i < 5; i++) {
                for (auto k = 0; k < test_item_data.size(); k++) {
                    auto e = test_item_data.at(k);
                    int random_number = 0;
                    if (k % 3 != 0) {
                        std::uniform_int_distribution<int> distribution(0, 100);
                        random_number = distribution(generator);
                    }

                    input_data.push_back(e + random_number);
                }
            }
            cout << input_data.size() << ", ";
            auto d = Dist3(input_data, test_material, "", false);
            //d.scoring_sys.parameters = p;
            d.run();
            cout << d.get_avg_util_rate() << ", "<<"plan util rate";
            for (auto plan : d.solution) {
                cout << plan.get_util_rate() << ", ";
            }
            
        }

        cout << endl;
    }
}

void test_dist_skyline2() {
    auto d = Dist3(test_item_data, test_material, "", false);
    d.run();
    for (auto plan : d.solution) {
        cout << 1/plan.get_util_rate() << ", ";
    }
    cout <<test_item_data.size()<<"," << d.get_avg_util_rate() << ", " << "plan util rate";
}



int main() {
    test_dist_skyline();
    
    
    return 0;
}