
class TreeNode:
    def __init__(self,value,children=[]):
        self.value=value
        self.children=children
        self.alpha=float("-inf")
        self.beta=float("inf")
    
def alpha_beta(node,depth,alpha,beta,maximizing_player):
    global pruned_count
    if depth==0 or not node.children:  
        return node.value,[node.value]
    if maximizing_player:
        max_value=float("-inf")
        max_path=[]
        for child_node in node.children:
            child_value,child_path=alpha_beta(child_node,depth-1,alpha,beta,False)
            if child_value>max_value:
                max_value=child_value
                max_path=[node.value]+child_path
            alpha=max(alpha,max_value)
            if alpha>=beta:
                pruned_count+=len(child_node.children)+1 
                break
        return max_value,max_path
    else:
        min_value=float("inf")
        min_path=[]
        for child_node in node.children:
            child_value,child_path=alpha_beta(child_node,depth-1,alpha,beta,True)
            if child_value<min_value:
                min_value=child_value
                min_path=[node.value]+child_path
            beta=min(beta,min_value)
            if alpha>=beta:
                pruned_count+=len(child_node.children)+1
                break
        return min_value,min_path
game_tree=TreeNode(0,[
    TreeNode(1,[TreeNode(3),TreeNode(12)]),
    TreeNode(4,[TreeNode(8),TreeNode(2)])
])
  
pruned_count=0
optimal_value,optimal_path=alpha_beta(game_tree,2,float("-inf"),float("inf"),True)
print("Pruned Count : ",pruned_count)
print("Optimal value: ",optimal_value)
print("Optimal path: ",optimal_path)