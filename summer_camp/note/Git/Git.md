# Git

版本迭代

### 常规操作

创建本地仓库：`git init`

创建远程仓库：`在Github上创建`

克隆：`git clone <远程仓库ssh>`

关联远程仓库：`git remote add origin <远程仓库ssh>`

显示远程仓库的信息：`git remote -v`

掌握工作区状态：`git status`

查看修改内容：`git diff`

将文件添加到暂存区：`git add <文件名>`

将暂存区版本提交到版本库：`git commit -m “...”`

##### 版本丢弃修改

查看工作区与版本库里面最新版本的区别：`git dff HEAD --readme.txt`

把工作区的修改全部撤销/用版本库里的版本替换工作区的版本（无论工作区是修改还是删除都可以复原）：`git checkout --<file>`

**注：没有commit到版本库就删除的文件是无法恢复的**

删除文件：`git rm -r<file>`



### 分支管理

查看分支：`git branch`

创建分支：`git branch <name>`

切换分支：`git checkout <name>/git switch <name>`

创建并切换分支：`git checkout -b <name>/git switch -c <name>`

合并分支：`git merge <name>/git merge --no-ff <name>`

**注：前者历史无分支，后者历史有分支**

删除分支：`git branch -d <name>`

强行删除未合并分支：`git branch -D <name>`

推送分支：`git push origin <name>`

抓取远程新提交：`git pull`(**解决推送失败是因为远程分支更新了的情况**)

建立本地分支和远程分支的关联：`git branch --set-upstream branch-name origin/branch-name`(**本地和远程的分支名称最好一致**)

把分叉提交历史整理成直线：`git rebase`

##### bug分支

储藏分支：`git stash`

查看stash内容：`git stash list`

恢复并删除stash内容：`git stash pop`

把bug提交的修改复制到当前分支：`git cherry-pick <commit id>`

##### 

### 版本回退

查看从最近到最远的历史提交日志：`git log --pretty=oneline`

`HEAD`指向的版本是当前版本，在版本的历史间穿梭：`git reset --hard commit_id(或者HEAD^)`

把暂存区的修改撤销掉：`git reset HEAD<file>`

**注意使用`reset`回退版本之后若关闭了git命令窗口则无法找到回退之前的版本id，需使用:`git reflog`**



### 冲突解决

当Git无法自动合并分支时，就必须首先解决冲突。解决冲突后，再提交，合并完成。解决冲突就是把Git合并失败的文件手动编辑为我们希望的内容，再提交

查看分支合并图：`git log --graph`