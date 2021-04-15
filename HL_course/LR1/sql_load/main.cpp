#include <string>
#include <iostream>
#include <fstream>

#include <Poco/Data/MySQL/Connector.h>
#include <Poco/Data/MySQL/MySQLException.h>
#include <Poco/Data/SessionFactory.h>

#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>
#include <Poco/JSON/Parser.h>
#include <Poco/Dynamic/Var.h>

auto main(int argc, char *argv[]) -> int
{
    if (argc < 2)
        return 0;
    std::string host(argv[1]);
    std::cout << "connecting to:" << host << std::endl;
    Poco::Data::MySQL::Connector::registerConnector();
    std::cout << "connector registered" << std::endl;

    std::string connection_str;
    connection_str = "host=";
    connection_str += host;
    connection_str += ";user=mai_user;db=sql_test;password=maiforever";

    Poco::Data::Session session(
        Poco::Data::SessionFactory::instance().create(
            Poco::Data::MySQL::Connector::KEY, connection_str));

    try
    {
        Poco::Data::Statement create_stmt(session);
        create_stmt << "CREATE TABLE IF NOT EXISTS `Author` (`id` INT NOT NULL AUTO_INCREMENT,"
                    << "`login` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL UNIQUE,"
                    << "`first_name` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,"
                    << "`last_name` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,"
                    << "`age` INT NOT NULL,"
                    << "PRIMARY KEY (`id`),KEY `fn` (`first_name`),KEY `ln` (`last_name`));";
        create_stmt.execute();
        std::cout << "table created" << std::endl;

        Poco::Data::Statement truncate_stmt(session);
        truncate_stmt << "TRUNCATE TABLE `Person`;";
        truncate_stmt.execute();

        std::string json;
        std::ifstream is("../data.json");
        std::istream_iterator<char> eos;
        std::istream_iterator<char> iit(is);
        while (iit != eos)
            json.push_back(*(iit++));
        is.close();

        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);
        Poco::JSON::Array::Ptr arr = result.extract<Poco::JSON::Array::Ptr>();

        size_t i{0};
        for (i = 0; i < arr->size(); ++i)
        {
            Poco::JSON::Object::Ptr object = arr->getObject(i);
            std::string first_name = object->getValue<std::string>("first_name");
            std::string last_name = object->getValue<std::string>("last_name");
            int age = object->getValue<int>("age");
            std::string login = object->getValue<std::string>("login");
            Poco::Data::Statement insert(session);
            insert << "INSERT INTO Person (login,first_name,last_name,age) VALUES(?, ?, ?, ?)",
                Poco::Data::Keywords::use(login),
                Poco::Data::Keywords::use(first_name),
                Poco::Data::Keywords::use(last_name),
                Poco::Data::Keywords::use(age);

            insert.execute();
        }

        std::cout << "Inserted " << i << " records" << std::endl; 
       
    }
    catch (Poco::Data::MySQL::ConnectionException &e)
    {
        std::cout << "connection:" << e.what() << std::endl;
    }
    catch (Poco::Data::MySQL::StatementException &e)
    {

        std::cout << "statement:" << e.what() << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "exception:" << e.what() << std::endl;
    }
    return 1;
}