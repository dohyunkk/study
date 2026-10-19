# from typing import Dict, TypedDict
from typing import TypedDict

sample_dict : dict[str, str] = {
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원'
}

print(sample_dict)
#* {'name': '김말똥', 'age': '39', 'job': '회사원'}

########################################################
class Person(TypedDict):
    name : str
    age : int
    job : str


person1 = Person(name='홍길동', age='40', job='도적')
print(person1)
# {'name': '홍길동', 'age': '40', 'job': '도적'}

########################################################

person2 : Person={'name' : '바바','age' : '32', 'job':'바바리안'}
print(person2)
# {'name': '바바', 'age': '32', 'job': '바바리안'}