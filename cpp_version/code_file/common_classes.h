#ifndef COMMON_CLASSES_H
#define COMMON_CLASSES_H

#include <cassert>
#include <iostream>
#include <vector>
#include <variant>
#include <sstream>
#include <random>
using namespace std;
string gen_uuid(size_t length=8);
class POS {
public:
    double x;
    double y;

    POS(double x = 0, double y = 0);

    POS copy() const;

    POS operator-(const POS& other) const;
    POS operator-(double scalar) const;

    POS operator/(double scalar) const;

    POS operator*(double scalar) const;

    POS operator+(const POS& other) const;
    POS operator+(double scalar) const;

    bool operator==(const POS& other) const;

    bool operator>(const POS& other) const;

    bool operator<(const POS& other) const;

    bool operator>=(const POS& other) const;

    bool operator<=(const POS& other) const;

    string to_string();
};
POS operator+(double scalar, const POS& pos);
enum class TYPE {RECT,LINE,POS};
class Rect {
public:
    POS start;
    POS end;
    int ID;

    Rect(POS start, POS end, int ID = -1);
    Rect(float x1,float y1,float x2,float y2,int ID = -1);
    Rect(POS start,float x2,float y2,int ID = -1);
    Rect(float x1,float y1,POS end,int ID = -1);
    Rect(int ID = -1);

    Rect operator-(const POS& other) const;
    Rect operator-(float other) const;
    Rect operator+(const POS& other) const;
    Rect operator+(float other) const;
    Rect operator*(float other) const;

    POS center() const;
    POS topLeft() const;
    POS topRight() const;
    POS bottomLeft() const;
    POS bottomRight() const;
    POS size() const;

    double width() const;
    double height() const;
    double area() const;

    Rect transpose() const;
    Rect copy() const;

    bool operator==(const Rect& other) const;
    bool operator==(const TYPE& other) const;
    bool operator!=(const Rect& other) const;

    bool contains(const POS& pos) const;
    bool contains(const Rect& rect) const;
    Rect operator&(const Rect& other) const;

    string to_string();
};
class Container {
public:
    Rect rect;
    int plan_id;

    Container(Rect rect, int plan_id = -1);

    bool operator==(const Container& other) const;
    bool operator==(const Rect& other) const;
    bool operator==(const TYPE& other) const;
    bool operator!=(const Container& other) const;
    bool contains(const POS& pos) const;
    bool contains(const Rect& rect) const;
    bool contains(const Container& other) const;
};

class Item {
public:
    int ID;
    Rect size;
    POS pos;

    Item(int ID, Rect size, POS pos);

    bool operator==(const Item& other) const;
    bool operator!=(const Item& other) const;
    Rect get_rect();
    Item copy();
};

class ProtoPlan {
public:
    int ID;
    Rect material;
    vector<Item> item_sequence;
    vector<Container> remain_containers;

    ProtoPlan(int ID, Rect material, vector<Item> item_sequence, vector<Container> remain_containers);
    float get_util_rate();
    vector<Container> get_remain_containers();
};

class Algo{
public:
    vector<Item> items;
    Rect material;
    vector<ProtoPlan> solution;
    string task_id;
    Algo(vector<float> items, pair<float,float> material, string task_id="");
    float get_avg_util_rate();

};


#endif 