pragma solidity ^0.8.0;

contract Payment {
    
    struct User {
        uint id;
        string name;
    }
    event TxnStatus(bool status);

    uint num_users = 0;
    mapping(uint => bool) public visited;
    mapping(uint => uint) public parent;
    mapping(uint => uint) public keys_list;
    mapping(uint => User) public user_list;
    mapping(uint => mapping(uint => uint)) public jointAccount;
    //index i, j indicates balance of i in the account for i<->j
    
    function registerUser(uint userId, string memory name) public {
        require(user_list[userId].id==0, "Error - User already exists");
        user_list[userId] = User(userId, name);
        keys_list[num_users++] = userId;
    }
    
    function createAcc(uint userId1, uint userId2, uint amount) public {
        require(user_list[userId1].id!=0, "Error - User 1 does not exist");
        require(user_list[userId2].id!=0, "Error - User 2 does not exist");
        require(jointAccount[userId1][userId2] == 0, "Account already exists");

        jointAccount[userId1][userId2] = amount/2;
        jointAccount[userId2][userId1] = amount/2;
    }
    
    function sendAmount(uint userId1, uint userId2) public {
        require(user_list[userId1].id!=0, "Error - User 1 does not exist");
        require(user_list[userId2].id!=0, "Error - User 2 does not exist");

        for(uint i=0;i<num_users;i++){
            visited[keys_list[i]] = false;
            parent[keys_list[i]] = 0;
        }

        uint[] memory queue = new uint[](num_users);
        queue[0] = userId1;
        visited[userId1] = true;
        parent[userId1] = 0;
        bool break_out = false;
        uint head = 0;
        uint tail = 1;

        while(head<tail){
            uint node = queue[head];
            head++;

            for (uint i = 0; i < num_users; i++) {
                uint neighbor = keys_list[i];
                if (!visited[neighbor] && jointAccount[node][neighbor]+jointAccount[neighbor][node]>0 ) {
                    visited[neighbor] = true;
                    parent[neighbor] = node;
                    queue[tail] = neighbor;
                    tail++;
                    if(neighbor==userId2){
                        break_out = true;
                        break;
                    }
                }
            }
            if(break_out){
                break;
            }
        }
        delete queue;

        uint node1 = parent[userId2];
        uint node2 = userId2;

        while(node2!=userId1){
            if(jointAccount[node1][node2]<1){
                emit TxnStatus(false);
                return;   
            }
            node2 = node1;
            node1 = parent[node2];
        }

        node1 = parent[userId2];
        node2 = userId2;

        while(node2!=userId1){
            jointAccount[node1][node2] -= 1;
            jointAccount[node2][node1] += 1;
            node2 = node1;
            node1 = parent[node2];
        }
        emit TxnStatus(true);

    }
    
    function closeAccount(uint userId1, uint userId2) public {
        require(user_list[userId1].id!=0, "Error - User 1 does not exist");
        require(user_list[userId2].id!=0, "Error - User 2 does not exist");

        jointAccount[userId1][userId2] = 0;
        jointAccount[userId2][userId1] = 0;
    }



}
