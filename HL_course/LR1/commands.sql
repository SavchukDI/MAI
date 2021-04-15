use sql_test; 
show tables;
CREATE TABLE IF NOT EXISTS `Person` (`id` INT NOT NULL AUTO_INCREMENT,`login` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NULL UNIQUE,`first_name` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,`last_name` VARCHAR(256) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,`age` INT NOT NULL,PRIMARY KEY (`id`),KEY `fn` (`first_name`),KEY `ln` (`last_name`));
describe Person;
select * from Person where id<10;
select id,first_name,last_name from Person where id>=10 and id<20;
explain select * from Person where id=10;
explain select * from Person where first_name='Lynn';
explain select * from Person where last_name='Eddison';
explain select * from Person where age=26;
explain select * from Person where id>10 and id<20;
explain select * from Person where first_name='Carter%';
insert into Person (login,first_name,last_name,age) values ('test','Иван','Иванов','ivanov@yandex.ru',99);
SELECT LAST_INSERT_ID();
select * from Person where id=LAST_INSERT_ID();
delete from Person where id= 100001;
show index from Person;
drop index fn_ln on Person;
create index fn_ln using btree on Person(first_name,last_name);
drop index ln_fn on Person;
create index ln_fn using btree on Person(last_name,first_name);
explain format=json select * from Person where first_name='Elle%' and last_name='A%';