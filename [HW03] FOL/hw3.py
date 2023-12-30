import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    # raise RuntimeError("You need to write this part!")

    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    i = 0
    for val in standardized_rules.values(): 
      isVarExist = False # proposition에 variable이 있으면 체크해준다.
      var = "x" + f"{i:0>4}" # 네 자리 정수 형태로 variable string을 생성
      for k, v in val.items(): 
        if k == "antecedents": # if key is antecedents
          if len(v) == 0: # meaning that this is triple
            break
          for lst in v: 
            for j in range(len(lst)-1):
              if lst[j] == "something" or lst[j] == "someone":
                lst[j] = var # something or someone 을 이전에 생성한 variable로 바꾸기
                isVarExist = True
                
        else: # consequent
          for j in range(len(v)-1):
            if v[j] == "something" or v[j] == "someone":
              v[j] = var
              isVarExist = True

      if isVarExist == True:
        i += 1
        variables.append(var)

    
    return standardized_rules, variables

def unify_var(var, x, theta): # 새로운 unification을 찾는 함수
  '''
  inputs: 
    var, a variable
    x, a variable or a constant
    theta, subs built so far in dictionary format
  '''
  if var in theta and theta[var] == x: 
    return None
  elif x in theta and theta[x] == var:
    return None
  else:
    return (var, x) # new x and subs[x] to be added



def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    for i in range(len(query)):
      if query[i] not in variables and datum[i] not in variables: # query와 datum 둘 다 variable이 아니면 
        if query[i] != datum[i]: # 이 둘이 다르면, 무조건 unify 할 수 없다. (predicate이 다르거나, boolean 값이 다른 경우, etc)
          return None, None

    # deepcopy
    Q = copy.deepcopy(query)
    D = copy.deepcopy(datum)
    subs = {}

    while(True): # iterate until all the substitutions are done
      pair = None # if new unification is added, this pair variable has that unification pair. Otherwise, None
      for i in range(len(Q)):
        if Q[i] not in variables and D[i] not in variables: # No need to unify the not variable 
          continue

        if Q[i] in variables: # query의 i번째 단어가 variable이면
          ret = unify_var(Q[i], D[i], subs) # unify_var 의 첫번째 인자로 Q[i]
        elif D[i] in variables: # datum의 i번째 단어가 variable이면
          ret = unify_var(D[i], Q[i], subs) # unify_var 의 첫번째 인자로 D[i]
        
        if ret != None: # unify_var 의 return 값이 None이 아니면, 새로운 unification이 만들어졌다는 뜻
          pair = ret # 해당 unification 을 pair에 대입
          # if new (key,val) is added, then all variable 'key' in query needs to be substituted to val first
          break # if new unifcation is added, before iterating to the end, we need to substitute the variable with the new set first

      if pair != None: # if new unification is returned 
        key, val = pair[0], pair[1]
        for i in range(len(Q)): 
          if Q[i] == key: # 새로운 unification에 맞게 variable을 unify 해준다.
            Q[i] = val
        subs[pair[0]] = pair[1] # add new 'x' : subs[x] to subs dictionary

      else: # nothing to be replaced, so done with substitution
        unification = Q 
        return unification, subs # unification succeeds, no more unification to be added. so terminate.

    return None, None # otherwise, unification fails

    '''

    raise RuntimeError("You need to write this part!")

    return unification, subs
    '''

def substitute(pro, subs, variables): # 해당 명제 안에 있는 variable을 현재의 unification set에 있는 값으로 교체
  proposition = copy.deepcopy(pro)
  for i, entry in enumerate(proposition):
    if entry in variables:
      if entry in subs:
        proposition[i] = subs[entry]
  return proposition # return new substituted proposition

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    R = copy.deepcopy(rule)
    G = copy.deepcopy(goals)
    applications = []
    goalsets = []

    for i, pro in enumerate(G): # fetch each proposition from goals
      uni, subs = unify(R["consequent"], pro, variables) # check if unification is possible for each goal with rule's consequent
      if uni == None and subs == None: # if unification impossible
        continue
      new_dict = {}
      new_dict["antecedents"] = [] 
      new_dict["consequent"] = uni # add unified consequent to applications

      new_goal = copy.deepcopy(G)
      del new_goal[i] # delete the goal that unified with applications[i]["consequent"]
        
      for ant in R["antecedents"]: # add unified antecedents to applications
        if bool(subs) == False: # if the subs dictionary is empty, it means that there was no variable in the consequent. 
          new_dict["antecedents"].append(ant) # add an unifed antecedent to the dictionary
          new_goal.append(ant) # add an unified antecedent to the new_goal
        else:
          unified_ant = substitute(ant, subs, variables) # unify the antecedents
          new_dict["antecedents"].append(unified_ant)
          new_goal.append(unified_ant) # add unified antecedents to new_goal

      applications.append(new_dict)
      goalsets.append(new_goal)
    
    return applications, goalsets

    '''

    # raise RuntimeError("You need to write this part!")

    return applications, goalsets
    '''

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''

    Q = []
    Q.append(copy.deepcopy(query))
    T = {}
    for key, r in rules.items(): # preprocess the triples first
      if key[0] == 't': # if key is the triple
        if r["consequent"][:3] == Q[0][:3]: # if the query before the boolean is equal to triple's consequent
          if r["consequent"][3] != Q[0][3]: # if the boolean is different
            return None # if all the words other than boolean value is same, but only the boolean is different, then it means the statement is false
          else:
            return True # if two statements are same, immediately return True
        continue 
      else: # if key is the rule
        if r["consequent"] == Q[0]: # if rule's consequent is exactly same with the query
          return True


    for key, r in rules.items():
      if key[0] == 't': # meaning that the key is triple     
        continue

      if r["consequent"] == Q[0]: # if rule's consequent is exactly same with the query
        continue

      applications, goalsets = apply(r, Q, variables)
      if len(goalsets) == 0: # meaning that this rule has nothing to do with the current goal
        continue
      succeed = []
      for goal in goalsets:
        for g in goal:
          succeed.append(backward_chain(g, rules, variables)) # all the antecedents should be True in order to make the consequent true
      
      true_cnt = 0
      for i in range(len(succeed)): # count the number of Trues
        if succeed[i] == True:
          true_cnt += 1

      if true_cnt == len(succeed): # if the length of the list is equal to the number of Trues, it means the query is also true
        return True
        
    return None
    # raise RuntimeError("You need to write this part!")
    # return proof
