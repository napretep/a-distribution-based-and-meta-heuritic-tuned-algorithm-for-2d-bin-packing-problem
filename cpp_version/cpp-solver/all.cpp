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
using RectAsVector = vector<float>;
using PlanPartAsVector = vector<RectAsVector>;
using ItemSequenceAsVector = PlanPartAsVector;
using RemainContainersAsVector = PlanPartAsVector;
using PlanAsVector =pair<ItemSequenceAsVector,RemainContainersAsVector>;
using PlanPackingLog = vector<PlanAsVector>;
using SolutionPackingLog = vector<PlanPackingLog>;
using SolutionAsVector = vector<PlanAsVector>;

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
    int ID;
    // Constructor
    Rect(POS start, POS end, int ID = -1) : start(start), end(end), ID(ID) {
        if (start >= end) {
            POS new_end = start;
            start = end;
            end = new_end;
        }
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
    vector<float> as_vector(){
        return {start.x, start.y, end.x, end.y};    
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
    Container(Rect rect, int plan_id = -1) {
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

};

//void print_containers(vector<Container> containers) {
//    
//
//
//}

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
    ProtoPlan(int ID, Rect material, vector<Item> item_sequence=vector<Item>(), vector<Container> remain_containers=vector<Container>()) {
        this->ID = ID;
        this->material = material;
        this->item_sequence = item_sequence;
        this->remain_containers = remain_containers;
    }
    float get_util_rate()const {
        double total_area = 0.0;
        for (const auto& item : item_sequence) {
            total_area += item.size.area();
        }
        double ratio = total_area / material.area();
        return ratio;
    }

    virtual vector<Container> get_remain_containers(bool throw_error = true) const {
        if (!remain_containers.empty()) {
            return remain_containers;
        }
        else {
            if(throw_error)
            {
                throw std::runtime_error("get_remain_containers must be implemented in a derived class when remain_containers is empty"); 
            }
            else {
                return remain_containers;
            }
        }
    }
    void print_remain_containers() {
        cout<<"plan_id="<<ID<<",remain_containers_count=" << remain_containers.size() << endl;
        for (const auto& container : get_remain_containers()) {
            cout << container.rect.to_string() << endl;
        }
    }
    PlanAsVector toVector() {
        ItemSequenceAsVector plan_item;
        for (auto item : item_sequence) {
            plan_item.push_back(item.get_rect().as_vector());
        }
        auto containers = get_remain_containers(true);
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
    Algo(vector<float> flat_items, pair<float, float> material, string task_id = "",bool is_debug=false):is_debug(is_debug) {
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
        for(auto plan: this->solution){
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

class Dist;

class ScoringSys {
public:

    Dist* algo;
    int item_sorting_param_count = 4;
    int container_scoring_param_count = 14;
    int gap_merging_param_count = 9;
    vector<float> parameters;
    ScoringSys(Dist& algo);

    vector<float> container_scoring_parameters()const {
        vector<float> p(this->parameters.begin(), this->parameters.begin() + container_scoring_param_count);
        return p;
    }
    vector<float>item_sorting_parameters()const {
        vector<float> p(this->parameters.begin() + container_scoring_param_count, this->parameters.begin()+container_scoring_param_count+ item_sorting_param_count);
        return p;
    }
    vector<float>gap_merging_parameters()const {
        vector<float> p(this->parameters.begin() + container_scoring_param_count + item_sorting_param_count, this->parameters.end());
        return p;
    }
    float item_sorting(float item_width, float item_height)const;
    float pos_scoring(Item item, Container container)const;
    float pos_scoring(float item_width, float item_height, float  container_begin_x, float  container_begin_y, float  container_width, float  container_height, float  plan_id)const;
    float gap_scoring( float current_plan_id, float current_max_len, float current_min_len, Container new_container, Container old_container)const;
};


class Dist :public Algo {

public:
    ScoringSys* scoring_sys;
    int current_item_idx;

    Dist(vector<float> items, pair<float, float> material = { 2440,1220 }, string task_id = "",bool is_debug=false) :
        Algo(items, material, task_id,is_debug)
    {
        this->scoring_sys = new ScoringSys(*this);
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

                    std::optional<Container> new_BR_corner = Container(Rect(new_rect.bottomRight(), POS(remove_rect.topRight().x, new_rect.topRight().y)), best_score.plan_id);
                    std::optional<Container> new_top_corner = Container(Rect(new_rect.topLeft(), remove_rect.topRight()), best_score.plan_id);
                    if ((new_BR_corner.value().rect.end.y - new_BR_corner.value().rect.start.y) < current_minL) {
                        new_BR_corner.reset();
                    }
                    if ((new_top_corner.value().rect.end.y - new_top_corner.value().rect.start.y) < current_minL) {
                        if (new_BR_corner.has_value()) {
                            new_BR_corner.value().rect.end = new_top_corner.value().rect.end;
                            new_top_corner.reset();
                        }
                    }
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
                        PlanPackingLog plan_packing_log;
                        plan_packing_log.push_back(new_plan.toVector());
                        this->packinglog.push_back(plan_packing_log);
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

                        

                        auto plancopy = plan;
                        this->packinglog.at(best_score.plan_id).push_back(plancopy.toVector());
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
                    float score = scoring_sys->pos_scoring(item_to_append, container);
                    score_list.push_back(ItemScore(item_to_append, container, score));
                }
                auto itemT = item.transpose();
                if (container.contains(itemT.size + corner_start)) {
                    auto item_to_append = itemT.copy();
                    item_to_append.pos = corner_start.copy();
                    float score = scoring_sys->pos_scoring(item_to_append, container);
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
                float score = scoring_sys->pos_scoring(item_to_append, container);
                score_list.push_back(ItemScore(item_to_append, container, score));
            }
            auto itemT = item.transpose();
            if (container.contains(itemT.size + corner_start)) {
                auto item_to_append = itemT.copy();
                item_to_append.pos = corner_start;
                float score = scoring_sys->pos_scoring(item_to_append, container);
                score_list.push_back(ItemScore(item_to_append, container, score));
            }
        }
        return score_list;
    }

    vector<Item> sorted_items()const {
        std::vector<Item> temp_sorted_items = items;  

        std::sort(temp_sorted_items.begin(), temp_sorted_items.end(),
            [this](const Item& a, const Item& b) {
                return scoring_sys->item_sorting(a.size.width(), a.size.height()) < scoring_sys->item_sorting(b.size.width(), b.size.height());
            });

        return temp_sorted_items;
    };


    std::optional<Container> container_merge_thinking(float current_minL, float current_maxL, ProtoPlan& plan, optional<Container>& optional_compare_corner, Container& container) {
        auto compare_corner = optional_compare_corner.value();
        auto result = container.rect & compare_corner.rect;
        bool merged = false;
        float merge_gap = this->scoring_sys->gap_scoring(float(plan.ID), current_maxL, current_minL, compare_corner, container);
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
ScoringSys::ScoringSys(Dist& algo)
{
    this->parameters = vector(this->container_scoring_param_count + this->container_scoring_param_count + this->gap_merging_param_count,1.0f);
    this->algo = &algo;
}
float ScoringSys::item_sorting(float item_width, float item_height)const {
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
float ScoringSys::pos_scoring(Item item, Container container)const {
    Rect cr = container.rect;
    Rect ir = item.size;
    return this->pos_scoring(ir.width(), ir.height(), cr.start.x, cr.start.y, cr.width(), cr.height(), container.plan_id);
}
float ScoringSys::pos_scoring(float item_width, float item_height, float  container_begin_x, float  container_begin_y, float  container_width, float  container_height, float  plan_id)const {
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
float ScoringSys::gap_scoring( float current_plan_id, float current_max_len, float current_min_len, Container new_container, Container old_container) const {
    auto cutting_rect = Rect(new_container.rect.bottomLeft(), POS(old_container.rect.bottomLeft().x, new_container.rect.topLeft().y));
    vector<float> X = {
        float(this->algo->current_item_idx) / this->algo->items.size(),
        current_plan_id / this->algo->solution.size(),
        cos((new_container.rect.start.x - old_container.rect.start.x) / current_max_len),
        min(new_container.rect.width(),new_container.rect.height()) / max(new_container.rect.width(),new_container.rect.height()),
        min(old_container.rect.width(),old_container.rect.height()) / max(old_container.rect.width(),old_container.rect.height()),
        cos((current_max_len - current_min_len) / this->algo->material.width()),
        1-new_container.rect.height()/ current_max_len,
        cos(cutting_rect.area()/(current_max_len*current_min_len)),
        min(cutting_rect.width(),cutting_rect.height())/ max(cutting_rect.width(),cutting_rect.height())
    };
    auto p = this->gap_merging_parameters();
    return std::inner_product(X.begin(), X.end(), p.begin(), 0.0);
};
//-------------------------------MaxRect algo-------------------------------------------------------------
class MaxRect :public Algo {
public:
    MaxRect(vector<float> flat_items, pair<float, float> material = { 2440,1220 }, string task_id = "",bool is_debug=false) :Algo(flat_items, material, task_id,is_debug) {
        
    }
    void run() {
        solution.clear();

        while (items.size()>0)
        {
            auto scores_li = find_possible_item_candidates();
            auto best_score = *std::min_element(scores_li.begin(), scores_li.end(), [](const ItemScore& p1, const ItemScore& p2) {
                return p1.getScore() < p2.getScore();
            });
            
            if(best_score.plan_id==-1){
               auto plan = ProtoPlan(solution.size(), material);
               plan.item_sequence.push_back(best_score.item);
               auto item_rect = best_score.item.get_rect();
               auto container_top = Container(Rect(item_rect.topLeft(), material.topRight()),plan.ID);
               auto container_btm = Container(Rect(item_rect.bottomRight(), material.topRight()),plan.ID);
               plan.remain_containers.push_back(container_top);
               plan.remain_containers.push_back(container_btm);
               solution.push_back(plan);
               PlanPackingLog planlog;
             
               planlog.push_back(plan.toVector());
               this->packinglog.push_back(planlog);
            //    plan.print_remain_containers();
            }
            else{
                auto& plan = solution[best_score.plan_id];
                plan.item_sequence.push_back(best_score.item);
                auto new_item = best_score.item;
                auto container = best_score.container;
                auto new_rect = new_item.get_rect();
                update_remain_containers(new_item,plan);
                // plan.print_remain_containers();
                this->packinglog.at(best_score.plan_id).push_back(plan.toVector());
            }
            
            erase(items, best_score.item);
        }
    }
    vector<ItemScore> find_possible_item_candidates() {
        vector<ItemScore> scores_li;
        
        for (auto i = 0; i < items.size(); i++)
        {
            Item& item = items[i];
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
    void update_remain_containers(Item& item, ProtoPlan& plan){
        vector<Container>container_to_remove;
        vector<optional<Container>>container_to_append;
        //split intersect rect
        for(auto& free_c : plan.remain_containers){
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
            }
        }

    }
};



class Skyline:public Algo{
    
public:
    class Plan:public ProtoPlan{
    public:
        
        vector<Container> SkylineContainers;
        vector<Container> WasteMap;
        Plan(int ID, Rect material, vector<Item> item_sequence = vector<Item>(), vector<Container> remain_containers = vector<Container>()) :
        ProtoPlan(ID,material){};
        vector<Container> get_remain_containers(bool throw_error=true)const override{
            vector<Container> result;
            for(auto c:SkylineContainers){
                result.push_back(c);
            }
            for(auto c:WasteMap){
                result.push_back(c);
            }
            return result;
        }

    };
    enum class ContainerType {
        WasteMap,
        Skyline
    };
    class Score{
    public:
        Item item;
        pair<int,int> container_range;
        float score;
        pair<float,float>scores;
        int plan_id;
        Skyline::ContainerType type;
        Score(Item item, pair<int,int> container_range,  int plan_id, Skyline::ContainerType type,float score=0, pair<float,float>scores={0,0}):item(item),container_range(container_range),score(score),scores(scores),plan_id(plan_id),type(type){};
    };

    vector<Skyline::Plan>solution;
    Skyline(vector<float> flat_items, pair<float, float> material = { 2440,1220 }, string task_id = "", bool is_debug = false) :Algo(flat_items, material, task_id, is_debug) {}
    void run() {
        // sorting the items
        sort(this->items.begin(),this->items.end(),[](const Item& a, const Item& b){
            float min_side_a = min(a.size.width(), a.size.height());
            float min_side_b = min(b.size.width(), b.size.height());
            return min_side_a > min_side_b;
            });
        if (this->is_debug){
            cout<<"sorted_item:\n";
            for(auto item:items){
                cout<<item.get_rect().to_string()<<endl;
            }
        }
        
        solution.clear();// avoid some strange things;
        for(auto& new_item:items){ // load item one by one
            if (this->is_debug) {cout<<"solution size="<<solution.size()<<endl;}
            vector<Skyline::Score> scores; // init a score list;
            for(auto& plan:solution){
                for(auto i=0;i<plan.WasteMap.size();i++){
                    Rect waste_rect = plan.WasteMap.at(i).rect;
                    if(waste_rect.contains(new_item.size+waste_rect.start)){
                        auto item_to_append = new_item.copy();
                        item_to_append.pos=waste_rect.start.copy();
                        auto score = Score(item_to_append,{i,i+1},plan.ID,ContainerType::WasteMap,calc_wastemap_score(item_to_append,plan.WasteMap.at(i)));
                        scores.push_back(score);
                    }
                    auto itemT = new_item.transpose();
                    if(waste_rect.contains(itemT.size+waste_rect.start)){
                        auto item_to_append = itemT.copy();
                        item_to_append.pos=waste_rect.start.copy();
                        auto score = Score(item_to_append,{i,i+1},plan.ID,ContainerType::WasteMap,calc_wastemap_score(item_to_append,plan.WasteMap.at(i)));
                        scores.push_back(score);
                    }
                }
            }
            if(scores.size()==0){
                for(auto& plan:solution){
                    for(auto i=0;i<plan.SkylineContainers.size();i++){
                        Rect skyline_rect = plan.SkylineContainers.at(i).rect;
                        int idx = get_placable_area(new_item,i,plan.SkylineContainers);
                        if(idx>=0){
                            auto item_to_append = new_item.copy();
                            item_to_append.pos=skyline_rect.start.copy();
                            pair<float, float> score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers);
                            auto score = Score(item_to_append, { i,idx + 1 }, plan.ID, ContainerType::Skyline, 0, score_calc);
                            scores.push_back(score);
                    }
                    auto itemT = new_item.transpose();
                    idx = get_placable_area(itemT,i,plan.SkylineContainers);
                    if(idx>=0){
                        auto item_to_append = itemT.copy();
                        item_to_append.pos=skyline_rect.start.copy();
                        pair<float, float> score_calc = calc_skyline_score(item_to_append, i, idx, plan.SkylineContainers);
                        auto score = Score(item_to_append,{i,idx+1},plan.ID,ContainerType::Skyline,0,score_calc);
                        scores.push_back(score);
                    }
                }
                }
            }
            if(scores.size()==0){
                vector<Container> fake_container = {Container(Rect(this->material.start, this->material.end))};
                int idx = get_placable_area(new_item, 0, fake_container);
                if(idx >= 0){
                    auto item_to_append = new_item.copy();
                    auto score = Score(item_to_append,{0,1},-1,ContainerType::Skyline,0,calc_skyline_score(item_to_append,0, idx, fake_container));
                    scores.push_back(score);
                }
                auto itemT = new_item.transpose();
                idx = get_placable_area(itemT, 0, fake_container);
                if(idx >= 0){
                    auto item_to_append = itemT.copy();
                    auto score = Score(item_to_append,{0,1},-1,ContainerType::Skyline,0,calc_skyline_score(item_to_append,0, idx, fake_container));
                    scores.push_back(score);
                }


            }
            if(scores.size()==0){
                throw runtime_error("no possible item candidates");
            }
            Score best_score = *min_element(scores.begin(),scores.end(),[](const Score& a, const Score& b){
                if(a.type!=b.type){
                    throw runtime_error("a.type!=b.type");
                }
                if(a.plan_id==b.plan_id){
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
                auto plan = Plan(solution.size(),Rect(0,0,test_material.first,test_material.second));
                auto container_top = Container(Rect(best_score.item.size.topLeft(), POS(best_score.item.size.width(), test_material.second)),plan.ID);
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
                if(best_score.type==ContainerType::WasteMap){ 
                    auto& container =plan.WasteMap.at(best_score.container_range.first);
                    // generate the best split plan
                    pair<Container,Container> split_1 ={
                        Container(Rect(new_rect.topLeft(),container.rect.topRight()),best_score.plan_id),
                        Container(Rect(new_rect.bottomRight(),POS(container.rect.bottomRight().x,new_rect.topRight().y),best_score.plan_id))
                    };  
                    pair<Container,Container> split_2 ={
                        Container(Rect(new_rect.topLeft(),POS(new_rect.topRight().x,container.rect.topRight().y)),best_score.plan_id),
                        Container(Rect(new_rect.bottomRight(),POS(new_rect.topRight().x,container.rect.topRight().y),best_score.plan_id))
                    };  
                    auto split_1_area = max(split_1.first.rect.area(),split_1.second.rect.area());
                    auto split_2_area = max(split_2.first.rect.area(),split_2.second.rect.area());
                    
                    optional<Container> maybe_newC_top=nullopt;
                    optional<Container> maybe_newC_right=nullopt;
                    if(split_1_area>split_2_area){
                        maybe_newC_top = split_1.first;
                        maybe_newC_right = split_1.second;
                    }
                    else{
                        maybe_newC_top = split_2.first;
                        maybe_newC_right = split_2.second;
                    }
                    if (maybe_newC_top.has_value() && (maybe_newC_top.value() == TYPE::LINE or maybe_newC_top.value() == TYPE::POS)) {
                        maybe_newC_top.reset();
                    }
                    if(maybe_newC_right.has_value() && (maybe_newC_right.value() == TYPE::LINE or maybe_newC_right.value() == TYPE::POS)){
                        maybe_newC_right.reset();
                    }
                    // maintain the old wastemap, if the new could merge to old
                    for(auto& waste_c:plan.WasteMap){
                        if(!maybe_newC_top.has_value() && !maybe_newC_right.has_value()){
                            break;
                        }
                        if(maybe_newC_right.has_value()){
                            auto& newC_right = maybe_newC_right.value();
                            auto result = newC_right.rect & waste_c.rect;
                            if (result ==TYPE::LINE){
                                auto diff = result.end - result.start;
                                if(diff.x==0){
                                    if (waste_c.rect.bottomRight()==newC_right.rect.bottomLeft() and waste_c.rect.topRight()==newC_right.rect.topLeft()) {
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
                        if(maybe_newC_top.has_value()){
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
                    maybe_container_top = Container(Rect(new_rect.topLeft(), POS(new_rect.topRight().x,this->material.height())),plan.ID);
                    if (new_rect.bottomRight().x != last_c.rect.end.x) {
                        maybe_container_right = Container(Rect(POS(new_rect.bottomRight().x, last_c.rect.start.y), POS(last_c.rect.end.x,this->material.height())));
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
                            auto c = Container(Rect(waste_c.rect.bottomLeft(), POS(min(waste_c.rect.bottomRight().x,new_rect.bottomRight().x), new_rect.bottomRight().y)), plan.ID);
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
    pair<float,float> calc_skyline_score(Item item,int begin_idx,int end_idx,vector<Container>containers){
        if (this->is_debug)
            {cout<<"calc_skyline_score.begin_idx="<<begin_idx<<",end_idx="<<end_idx<<",containers.size"<<containers.size()<<endl;}

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
    }
    float calc_wastemap_score(Item item,Container container){
        if (item.size.height()>item.size.width()){
            return container.rect.width()-item.size.width();
        }
        else{
            return container.rect.height()-item.size.height();
        }
    }
    int get_placable_area(Item item,int begin_idx,vector<Container>containers){
        if (this->is_debug)
           { cout<<"get_placable_area.begin_idx="<<begin_idx<<",containers.size"<<containers.size()<<endl;}
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

int main() {
    auto d = Dist(test_item_data);
    d.run();
    return 0;
}