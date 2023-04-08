pragma solidity ^0.8.0;

contract Payment {
    
    struct User {
        uint id;
        string name;
    }
    
    mapping(uint => User) public user_list;
    mapping(uint => mapping(uint => uint)) public jointAccount;
    //index i, j indicates balance of i in the account for i<->j
    
    function registerUser(uint userId, string memory name) public {
        require(user_list[userId].id==0, "User already exists");
        user_list[userId] = User(userId, name);
    }
    
    function createAcc(uint userId1, uint userId2) public {
        require(user_list[userId1].id!=0, "User 1 does not exist");
        require(user_list[userId2].id!=0, "User 2 does not exist");
        require(jointAccount[userId1][userId2] == 0, "Account already exists");

        uint balance = 10;
        //uint balance = expRandom(10);
        jointAccount[userId1][userId2] = balance/2;
        jointAccount[userId2][userId1] = balance/2;
    }
    
    function sendAmount(uint userId1, uint userId2, uint amount) public {
        require(user_list[userId1].id!=0, "User 1 does not exist");
        require(user_list[userId2].id!=0, "User 2 does not exist");
        require(jointAccount[userId1][userId2] >= amount, "Insufficient balance for User 1");

        jointAccount[userId1][userId2] -= amount;
        jointAccount[userId2][userId1] += amount;
    }
    
    function closeAccount(uint userId1, uint userId2) public {
        require(user_list[userId1].id!=0, "User 1 does not exist");
        require(user_list[userId2].id!=0, "User 2 does not exist");

        jointAccount[userId1][userId2] = 0;
        jointAccount[userId2][userId1] = 0;
    }



}
