#include <iostream>
#include <stdio.h>
#include <vector>
#define max_length 300000
using namespace std;
int Answer;
void calculate(vector <int> v, vector <int> o);
int main(void)
{
	int T, test_case, num, temp;
	vector <int> num_case(max_length);
	vector <int> odd_num;
	vector<int>::iterator begin_iter2 = odd_num.begin();
	vector<int>::iterator end_iter2 = odd_num.end();
	vector<int>::iterator begin_iter = num_case.begin();
	vector<int>::iterator end_iter = num_case.end();
	/*
	   The freopen function below opens input.txt file in read only mode, and afterward,
	   the program will read from input.txt file instead of standard(keyboard) input.
	   To test your program, you may save input data in input.txt file,
	   and use freopen function to read from the file when using scanf function.
	   You may remove the comment symbols(//) in the below statement and use it.
	   But before submission, you must remove the freopen function or rewrite comment symbols(//).
	 */
	 // freopen("input.txt", "r", stdin);

	 /*
		If you remove the statement below, your program's output may not be rocorded
		when your program is terminated after the time limit.
		For safety, please use setbuf(stdout, NULL); statement.
	  */
	cin >> T;
	for (test_case = 0; test_case < T; test_case++)
	{	

		cin >> num;
		for (int i = 0; i < num; i++) {
			cin >> temp;
			num_case[temp] += 1;
		}
		//for (int i = 0; i < num; i++) {
		//	cin >> temp;
		//	num_case.push_back(temp);
		//}
		calculate(num_case,odd_num);
		//scanf_s("%[^\n]s", temp, max_length);
		/////////////////////////////////////////////////////////////////////////////////////////////
		/*
		   Implement your algorithm here.
		   The answer to the case will be stored in variable Answer.
		 */
		 /////////////////////////////////////////////////////////////////////////////////////////////

		 // Print the answer to standard output(screen).
		printf("Case #%d\n", test_case + 1);
		printf("%d\n", Answer);

	}

	return 0;//Your program should return 0 on normal termination.
}

void calculate(vector <int> v, vector <int> o) {
 // 각 넘버가 몇개씩인지 계산하고, 홀수인것만 꺼내서 계산
	int num = 0;
	for (vector<int>::iterator iter = v.begin(); iter != v.end(); iter++) {
		if (*iter % 2 == 1) {
			//cout << "값: " <<*iter<<"?:"<< find(v.begin(), v.end(), *iter) - v.begin() << endl;
			//o.push_back(find(v.begin(), v.end(), *iter) - v.begin()); // 인덱스값
			o.push_back(num); // 인덱스값
		}
		num++;
	}

	for (vector<int>::iterator iter = o.begin(); iter != o.end(); iter++) {
		//cout << "값2: " << *iter << endl;
		Answer ^= *iter;
		
	}

}
