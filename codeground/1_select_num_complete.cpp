#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
#define max_length 3000000
using namespace std;
int Answer;
void calculate(map <int, int> m, vector <int> o);
int main(void)
{
	int T, test_case, num, temp;
	vector <int> odd_num;
	vector<int>::iterator begin_iter2 = odd_num.begin();
	vector<int>::iterator end_iter2 = odd_num.end();
	map<int, int> m;
    cin >> T;
	int tmp = 0;
	for (test_case = 0; test_case < T; test_case++)
	{	

		cin >> num;
		for (int i = 0; i < num; i++) {
			cin >> temp;
			if (m.find(temp) == m.end()) {
				m.insert(make_pair(temp,1));
			}
			else {
				tmp = m[temp];
				m[temp] = tmp + 1;
			}
		}
		calculate(m,odd_num);
		printf("Case #%d\n", test_case + 1);
		printf("%d\n", Answer);

	}

	return 0;//Your program should return 0 on normal termination.
}

void calculate(map <int,int> m, vector <int> o) {
	for (map<int,int>::iterator iter = m.begin(); iter != m.end(); iter++) {
		if (iter->second % 2 == 1) {
			o.push_back(iter->first); // 인덱스값
		}
	}

	for (vector<int>::iterator iter = o.begin(); iter != o.end(); iter++) {
		Answer ^= *iter;
	}

}
