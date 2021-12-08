#include <vector>
#include <iostream>
#include <memory>
using namespace std;

class Foo {
public:
    Foo(int n): num(n){}
    void print() {
        cout << num << endl;
    }
    ~Foo() {
        cout << "Foo is destroyed!" << endl;
    }
private:
    int num;
};

int main() {
    auto a = make_shared<Foo>(2);
    
    cout << "begin" << endl;
    {
        vector<Foo> vec{*a};
        cout << "vec is going to end" << endl;
    }
    a->print();
    cout << "end" << endl;
    return 0;
}