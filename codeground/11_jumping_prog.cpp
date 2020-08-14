#include <iostream>

using namespace std;

int Answer;
template <typename T>
struct NODE {
public:
	T value;
	struct NODE<T>* next = nullptr;
};
template <typename T>
class Linked_List {
private:
	NODE<T>* head;
	NODE<T>* tail;
	NODE<T>* before;
	int size = 0;
public:
	Linked_List() : head(nullptr), tail(nullptr) {}
	~Linked_List() {}
	void addNode(T _value) {
		NODE<T> * node = new NODE < T>;
		size++;
		node->value = _value;
		node->next = nullptr;
		if (head == nullptr) {
			head = node; 
			tail = node; 
		} 
		else {
			before = node;
			tail->next = node; 
			tail = tail->next; 
		}
	}
	int find(int len) {
		NODE<T>* temp = head;
		int point = 0;
		int spot = 0;
		while (head != tail) {
			//cout << "헤드값:"<< head->value << endl;
			if (head->next->value - head->value > len) {
				return -1;
			}
			if (head->next->value - spot <= len) {
				before = head;
				head = head->next;
			}
			else {
				//head = before;
				spot = head->value;
				point++;
			}
		}
		point++;
		return point;
	}
	void out() {
		NODE<T>* temp = head;
		cout << temp->value << endl;
		temp = tail;
		cout << temp->value << endl;
	}
};

int main(int argc, char** argv)
{
	int T, test_case, stone;
	cin >> T;
	for (test_case = 0; test_case < T; test_case++)
	{
		cin >> stone;
		int spot = 0;
		int max_jump = 0;
		Linked_List<int> list;
		list.addNode(spot);
		for (int i = 0; i < stone; i++) {
			cin >> spot;
			list.addNode(spot);
		}
		cin >> max_jump;
		//list.out();
		Answer = list.find(max_jump);
		
		
		cout << "Case #" << test_case + 1 << endl;
		cout << Answer << endl;
	}

	return 0;
}
