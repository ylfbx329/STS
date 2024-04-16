import os
import tkinter as tk
from tkinter import ttk

def populate_tree(tree, node):
    tree.delete(*tree.get_children(node))
    path = tree.item(node, "text")
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            child_node = tree.insert(node, "end", text=item, open=False)
            populate_tree(tree, child_node)
        else:
            tree.insert(node, "end", text=item)

root = tk.Tk()
root.title("文件浏览器")

tree = ttk.Treeview(root)
tree.heading("#0", text="文件系统", anchor='w')
tree.pack(fill="both", expand=True)

# 添加根节点
root_node = tree.insert("", "end", text=os.getcwd(), open=False)
populate_tree(tree, root_node)

root.mainloop()
