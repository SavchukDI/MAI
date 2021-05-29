#include <iostream>
#include <fstream>
#include <regex>

using namespace std;

int main() {

    size_t max_shards = 3;
    size_t shard_num;

    std::string line;
    std::ifstream infile("data/data.sql");
    std::ofstream outfile("data/data_sh.sql");

    std::string data_shard_0 = "use sql_test;\nINSERT INTO Person (login, first_name,last_name,age) VALUES\n";
    std::string data_shard_1 = "use sql_test;\nINSERT INTO Person (login, first_name,last_name,age) VALUES\n";
    std::string data_shard_2 = "use sql_test;\nINSERT INTO Person (login, first_name,last_name,age) VALUES\n";

    std::smatch match;
    std::regex pattern("([0-9]{3}-[0-9]{2}-[0-9]{4})");

    while (std::getline(infile, line)) {
        if (std::regex_search(line, match, pattern)) {
            shard_num = std::hash<std::string>{}(match[0]) % max_shards;
            switch (shard_num) {
                case 0:
                    data_shard_0 += line +'\n';
                    break;
                case 1:
                    data_shard_1 += line +'\n';
                    break;
                case 2:
                    data_shard_2 += line +'\n';
                    break;
            }
        };
    }

    infile.close();

    data_shard_0.pop_back();
    data_shard_0.pop_back();
    data_shard_0 += "\n -- sharding:0\n;\n";
    data_shard_1.pop_back();
    data_shard_1.pop_back();
    data_shard_1 += "\n -- sharding:1\n;\n";
    data_shard_2.pop_back();
    data_shard_2.pop_back();
    data_shard_2 += "\n -- sharding:2\n;\n";

    outfile << data_shard_0 << data_shard_1 << data_shard_2 << std::endl;
    outfile.close();
    return 0;
}