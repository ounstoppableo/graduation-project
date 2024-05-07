import ast,_ast
import re
from packages.getNodeMetric import customast
import astunparse
# import yaml

# stream = open("config",'r')
# config = yaml.load(stream)

class MyAst(ast.NodeVisitor):
    def __init__(self):
        self.fileName = None
        self.defmagic = set()
        self.usedmagic = set()
        self.subscriptnodes = [] #avoid replicate node code smell reports
        self.containernodes = [] #avoid replicate node code smell reports
        self.messagenodes = [] #avoid replicate node code smell reports
        self.scopenodes = []
        self.imports = set()
        self.result = []
        self.current_function = ''
        self.current_class = ''

    def count_lines(self,node):
        childnodes = list(ast.walk(node))
        lines = set()
        for n in childnodes:
          if hasattr(n,'lineno'):
            lines.add(n.lineno)
        return len(lines)

    def visit_TryExcept(self,node):
        exceptions = ["BaseException","Exception","StandardError"]
        i = len(node.handlers)
        for item in node.handlers:
            i = i-1
            if i!=0 and astunparse.unparse(item.body[0]).strip() == "pass":
                self.result.append((7,self.fileName,node.lineno,'pass'))
                self.generic_visit(node) 
                return
            if item.type is not None:
                if isinstance(item.type,_ast.Tuple) or isinstance(item.type,_ast.List):
                    for e in item.type.elts:
                        if hasattr(e,"id") and e.id in exceptions:
                            self.result.append((7,self.fileName,node.lineno,'genrallist'))
                            self.generic_visit(node) 
                            return
                elif i!=0 and hasattr(item.type,"id") and item.type.id in exceptions:
                    self.result.append((7,self.fileName,node.lineno,'general'))
                    self.generic_visit(node) 
                    return
            elif i!=0:
              self.result.append((7,self.fileName,node.lineno,'general'))
              self.generic_visit(node)
              return

    def visit_ClassDef(self,node):
      # baseClassesSize
      className = node.name
      self.current_class = className

      lines = set()
      res = [node]
      while len(res) >= 1:
        t = res[0]
        for n in ast.iter_child_nodes(t):
          if not hasattr(n,'lineno') or ((isinstance(t,_ast.FunctionDef) or isinstance(t,_ast.ClassDef)) and n == t.body[0] and isinstance(n,_ast.Expr)):
            continue
          lines.add(n.lineno)
          if isinstance(n,_ast.ClassDef) or isinstance(n,_ast.FunctionDef):
            continue
          else:
            res.append(n)
        del res[0]

      #CLOC
      # lines = set()
      # res = [node]
      # while len(res) >= 1:
      #   t = res[0]
      #   for n in ast.iter_child_nodes(t):
      #     if not hasattr(n,'lineno') or ((isinstance(t,_ast.FunctionDef) or isinstance(t,_ast.ClassDef)) and n == t.body[0] and isinstance(n,_ast.Expr)):
      #       continue
      #     lines.add(n.lineno)
      #     if isinstance(n,_ast.ClassDef) or isinstance(n,_ast.FunctionDef):
      #       continue
      #     else:
      #       res.append(n)
      #   del res[0]
      end_line = node.body[-1].end_lineno
      self.result.append(('CLOC','CLASS',className,self.fileName,node.lineno,end_line - node.lineno + 1))

      #get NOA and NOM
      NOA = 0
      NOM = 0
      for item in node.body:
            # if node is assign
            if isinstance(item, ast.Assign):
                NOA += 1
            # or is def
            if isinstance(item,ast.FunctionDef):
                NOM+=1
      self.result.append(('NOA','CLASS',className,self.fileName,node.lineno,NOA))
      self.result.append(('NOM','CLASS',className,self.fileName,node.lineno,NOM))

      self.generic_visit(node)
      self.current_class = ''

    def visit_FunctionDef(self,node):
      # argsCount
      def findCharacter(s,d):
        try:
          value = s.index(d)
        except ValueError:
          return -1
        else:
          return value
      funcName = node.name.strip() if not self.current_class else self.current_class+'.'+node.name.strip()
      p = re.compile("^(__[a-zA-Z0-9]+__)$")
      if p.match(funcName.strip()) and funcName != "__import__" and funcName != "__all__":
        self.defmagic.add((funcName,self.fileName,node.lineno))
      stmt = astunparse.unparse(node.args)
      arguments = stmt.split(",")
      argsCount = 0
      for element in arguments:
        if findCharacter(element,'=') == -1:
          argsCount += 1
      if(argsCount!=0):
        self.result.append(('PAR','DEF',funcName,self.fileName,node.lineno,argsCount))
      #function length
      # lines = set()
      # res = [node]
      # while len(res) >= 1:
      #   t = res[0]
      #   for n in ast.iter_child_nodes(t):
      #     if not hasattr(n,'lineno') or ((isinstance(t,_ast.FunctionDef) or isinstance(t,_ast.ClassDef)) and n == t.body[0] and isinstance(n,_ast.Expr)):
      #       continue
      #     lines.add(n.lineno)
      #     if isinstance(n,_ast.ClassDef) or isinstance(n,_ast.FunctionDef):
      #       continue
      #     else:
      #       res.append(n)
      #   del res[0]
      end_line = node.body[-1].end_lineno
      self.result.append(('MLOC','DEF',funcName,self.fileName,node.lineno,end_line - node.lineno + 1)) 
      #nested scope depth
      if node in self.scopenodes:
        self.scopenodes.remove(node)
        self.generic_visit(node)
        return
      dep = [[node,1]] #node,nestedlevel
      maxlevel = 1
      while len(dep) >= 1:
        t = dep[0][0]
        currentlevel = dep[0][1]
        for n in ast.iter_child_nodes(t):
          if isinstance(n,_ast.FunctionDef):
            self.scopenodes.append(n)
            dep.append([n,currentlevel+1])
        maxlevel = max(maxlevel,currentlevel)
        del dep[0]
      if maxlevel>1:
        self.result.append(('DOC','DEF',funcName,self.fileName,node.lineno,maxlevel)) #DOC
    
      self.generic_visit(node)

    def visit_Call(self,node):
      funcName = astunparse.unparse(node.func).strip()
      p = re.compile("^(__[a-zA-Z0-9]+__)$")
      if p.match(funcName) and funcName != "__import__" and funcName != "__all__":
        self.usedmagic.add((funcName,self.fileName,node.lineno))  

      self.generic_visit(node)

    def visit_IfExp(self,node):
      expr = astunparse.unparse(node)
      exprLength = len(expr.strip()) - expr.count(' ') - 2
      childnodes = list(ast.walk(node))
      lines = 0
      for n in childnodes:
        if hasattr(n,'lineno'):
          lines = max(n.lineno-node.lineno,lines)
      lines = lines + 1
      self.result.append((10,self.fileName,node.lineno,exprLength,lines))
      self.generic_visit(node) 

    def visit_Subscript(self,node):
      if node in self.subscriptnodes:
        self.subscriptnodes.remove(node)
        self.generic_visit(node)
        return
      maxcount = 1
      t = node
      while True:
        if isinstance(t.value,_ast.Subscript):
          self.subscriptnodes.append(t.value)
          maxcount = maxcount + 1
          t = t.value
        else:
          break
      self.result.append((6,self.fileName,node.lineno,maxcount)) #LEC
      self.generic_visit(node) 

    def visit_Attribute(self,node):
      if node in self.messagenodes:
        self.messagenodes.remove(node)
        self.generic_visit(node)
        return
      maxcount = 1
      t = node
      while True:
        if isinstance(t.value,_ast.Attribute):
          self.messagenodes.append(t.value)
          maxcount = maxcount + 1
          t = t.value
        else:
          break
      if maxcount>1:
        self.result.append((13,self.fileName,node.lineno,maxcount)) #LMC
      self.generic_visit(node)

    def visit_Import(self,node):
      if self.fileName[-12:] == '\\__init__.py':
        self.generic_visit(node)
        return
      for alias in node.names:
        if len(alias.name)>4 and alias.name[0:2] == '__' and alias.name[-2:] == '__':
            continue
        if alias.asname is not None:
          for (name,file,lineno) in self.imports:
            if name==alias.asname and self.fileName==file:
              break
          else:
            self.imports.add((alias.asname,self.fileName,node.lineno))
        elif alias.name != '*':
          for (name,file,lineno) in self.imports:
            if name==alias.name and self.fileName==file:
              break
          else:
            self.imports.add((alias.name,self.fileName,node.lineno))
      self.generic_visit(node)

    def visit_ImportFrom(self,node):
      if self.fileName[-12:] == '\\__init__.py':
        self.generic_visit(node)
        return
      try:
        if node.module is not None and len(node.module)>4 and node.module[0:2] == '__' and node.module[-2:] == '__':
          self.generic_visit(node)
          return
      except:
        print (astunparse.unparse(node))
      for alias in node.names:
        if len(alias.name)>4 and alias.name[0:2] == '__' and alias.name[-2:] == '__':
            continue
        if alias.asname is not None:
          for (name,file,lineno) in self.imports:
            if name==alias.asname and self.fileName==file:
              break
          else:
            self.imports.add((alias.asname,self.fileName,node.lineno))
        elif alias.name != '*':
          for (name,file,lineno) in self.imports:
            if name==alias.name and self.fileName==file:
              break
          else:
            self.imports.add((alias.name,self.fileName,node.lineno))
      self.generic_visit(node)



if __name__ == '__main__':
    pass