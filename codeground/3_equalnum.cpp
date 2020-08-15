#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
using namespace std;
int Answer;
int change(int num);
int main(void)
{
	int T, test_case,number;
	cin >> T;
	for (test_case = 0; test_case < T; test_case++)
	{
		//테스트 케이스 넘버
		cin >> number;
		Answer = change(number);

		printf("Case #%d\n", test_case + 1);
		printf("%d\n", Answer);

	}

	return 0;//Your program should return 0 on normal termination.
}
int change(int num) {
	// 기저 b를 조건에 의해서 1에서 n+1까지 고려해준다;
	int min = 0;
	vector<string> v;
	vector<string>::iterator iter;
	for (int i = 2; i < num + 2; i++) {
		int temp = num;
		while (temp != 0) {
			if (temp % 1 > 10) {
				v.push_back(to_string(temp % i));
				v.push_back(",");
			}
			else {
				v.push_back(to_string(temp % i));
			}
			temp = temp / i;
		}
		string check = v.front();
		//cout << "i:" <<i<< endl;
		while (v.size() != 0) {
			//cout << v.back() << endl;
			if(v.back() == ",") v.pop_back();
			if (check ==v.back()) {
				v.pop_back();
			}
			else {
				break;
			}
		}
		//for (iter = v.begin(); iter != v.end(); ++iter) {
		//		//중복이 되지 않는다면, clear해주고 for문을 탈출한다.
		//	if (check != *iter) {
		//		v.clear();
		//		break;
		//	}	
		//	else {
		//		check = *iter;
		//	}
		//}
		if (v.size() == 0) {
			min = i;
			break;
		}
		v.clear();
		if (i == num + 1) {
			min = i;
			break;
		}
	}
	return min;
}
