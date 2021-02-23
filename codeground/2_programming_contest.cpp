#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int Answer;
int main(int argc, char** argv)
{
	int T, test_case,score, max_standard_num,student_num,temp;
	
	cin >> T;
	for (test_case = 0; test_case < T; test_case++)
	{
        vector<int> student;
		Answer = 0;
		temp = 0;
		max_standard_num = 0;
		cin >> student_num;
		for (int i = 0; i < student_num; i++) {
			cin >> score;
			student.push_back(score);
		}
		sort(student.begin(),student.end());

		for (int i = 0; i < student_num; i++) {
			if (student[i]+ student_num-i > max_standard_num) {
				max_standard_num = student[i] + student_num - i;
			}
		}
		for (int i = 0; i < student_num; i++) {
			if (student[i] + student_num  >= max_standard_num) {
				Answer++;
			}
		}
		cout << "Case #" << test_case + 1 << endl;
		cout << Answer << endl;
	}

	return 0;//Your program should return 0 on normal termination.
}
