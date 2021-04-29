#include <gtest/gtest.h>
#include "config/config.h"
#include "database/database.h"
#include "database/person.h"
#include <Poco/Data/SessionFactory.h>


using Poco::Data::Session;
using Poco::Data::Statement;
using namespace Poco::Data::Keywords;

class TestApp : public ::testing::Test {
protected:
    TestApp() {
        Config::get().host() = "127.0.0.1";
        Config::get().database() = "sql_test";
        Config::get().port() = "3306";
        Config::get().login() = "mai_user";
        Config::get().password() = "maiforever";
    }
    ~TestApp() {
        Poco::Data::Session session = database::Database::get().create_session();
        Statement drop(session);
        drop << "DELETE FROM Person", now;
        Statement reset_ai(session);
        reset_ai << "ALTER TABLE Person AUTO_INCREMENT = 1", now;
    }
     void SetUp() {}
     void TearDown() {}

protected:
};

TEST(TestApp, TestPerson) {

    database::Person person;
    person.login() = "test_1";
    person.first_name() = "Ivan";
    person.last_name() = "Ivanov";
    person.age() = 44;

    testing::internal::CaptureStdout();
    person.save_to_mysql();
    ASSERT_EQ(testing::internal::GetCapturedStdout(), "inserted:1\n");

    person.login() = "test_2";
    person.first_name() = "Daniil";
    testing::internal::CaptureStdout();
    person.save_to_mysql();
    ASSERT_EQ(testing::internal::GetCapturedStdout(), "inserted:2\n");
    person.login() = "dan";
    person.last_name() = "sav";
    testing::internal::CaptureStdout();
    person.save_to_mysql();
    ASSERT_EQ(testing::internal::GetCapturedStdout(), "inserted:3\n");

    database::Person result = database::Person::read_by_login("test_1");
    ASSERT_EQ(result.get_first_name(), "Ivan");
    ASSERT_EQ(result.get_last_name(), "Ivanov");

    database::Person result2 = database::Person::read_by_login("dan");
    ASSERT_EQ(result2.get_first_name(), "Daniil");
    ASSERT_EQ(result2.get_last_name(), "Sav");

    auto results = database::Person::read_all();
    ASSERT_EQ(results.size(), 3);
    ASSERT_EQ(results.at(0).get_first_name(), "Dan");

    auto results2 = database::Person::search("Dan", "Sav");
    ASSERT_EQ(results2.size(), 3);
    ASSERT_EQ(results2.at(0).get_first_name(), "Dan");

    auto results3 = database::Person::search("Daniil", "Sav");
    ASSERT_EQ(results3.size(), 1);
    ASSERT_EQ(results3.at(0).get_login(), "test_2");

}


int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}