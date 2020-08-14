#include <iostream>
#include <string.h>
#include <string>
using namespace std;
int Answer;
int main(int argc, char** argv)
{
	int T, test_case,length;
	string num;
	
	cin >> T;
	for (test_case = 0; test_case < T; test_case++)
	{
		Answer = 0;
		cin >> length;
		int row = 0;
		int	col = 0;
		
		// 입력받은 한변의 길이 length에 맞춰서 2차원 배열 동적 할당을 해준다.
		// mirror는 0,1,2 거울의 정보를 담는 곳
		// mirror_check는 지나갔는지 아닌지 체크하는 곳
		string** mirror = new string * [length];
		bool ** mirror_check = new bool* [length];
		for (int i = 0; i < length; ++i) {
			mirror[i] = new string[length];
			mirror_check[i] = new bool[length];
		}
		// 인덱스에 거울 정보를 넣어준다
		for (int i = 0; i < length; i++) {
			cin >> num;
			for (int j = 0; j < length; j++) {
				mirror[i][j] = num[j];
				mirror_check[i][j] =false;
			}
		}
		/* 초기 빛의 뱡향은 R
			mirror 배열의 인덱스를 확인하면서 0, 1, 2 각 거울의 상태에 따라 나눠주고
			들어가는 빛의 방향에 따라서 case를 나눠준다.
			direction은 들어가는 빛의 방향이다.
			거울이 어떻게 반사하는지에 따라서 row값과 col값을 1을 더해주거나 빼준다.
			중복체크는 mirror_check인덱스에 true인지 false인지를 체크해서 값을 더해준다.
		*/
		string direction = "R";
		while (true) {
			if (mirror[row][col] == "0") {
				if (direction == "R") {
					col+=1;
				}
				else if (direction == "L") {
					col-=1;
				}
				else if (direction == "U") {
					row-= 1;
				}
				else if (direction == "D") {
					row+= 1;
				}
				if (row == length || col == length || row < 0 || col < 0) {
					break;
				}
			}
			else if (mirror[row][col] == "1") {
				if (!mirror_check[row][col]) {
					mirror_check[row][col] = true;
					Answer++;
;				}
				if (direction == "R") {
					row-= 1;
					direction = "U";
				}
				else if (direction == "L") {
					row+= 1;
					direction = "D";
				}
				else if (direction == "U") {
					col+= 1;
					direction = "R";
				}
				else if (direction == "D") {
					col-= 1;
					direction = "L";
				}
				if (row == length || col == length || row < 0 || col < 0) {
					break;
				}
			}
			else if (mirror[row][col] == "2") {
				if (!mirror_check[row][col]) {
					mirror_check[row][col] = true;
					Answer++;
				}
				if (direction == "R") {
					row+= 1;
					direction = "D";
				}
				else if (direction == "L") {
					row-= 1;
					direction = "U";
				}
				else if (direction == "U") {
					col-= 1;
					direction = "L";
				}
				else if (direction == "D") {
					col+= 1;
					direction = "R";
				}
				if (row == length || col == length || row < 0 || col <0) {
					break;
				}
			}
		}
		delete[] mirror_check;
		delete[] mirror;
		cout << "Case #" << test_case + 1 << endl;
		cout << Answer << endl;
	}

	return 0;
}
