from maze import Maze
############# Write Your Library Here ##########
from collections import deque, defaultdict
import heapq
import math
import itertools
################################################

def search(maze, func):
    return {
        "bfs": bfs,
        "ids":ids,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)

def bfs(maze:Maze):
    """
    [Problem 01] 제시된 stage1 맵 세 가지를 BFS를 구현하여 목표지점을 찾는 경로를 return하시오.
    """
    start_point=maze.startPoint()
    path=[]
    ####################### Write Your Code Here ################################
    n, m = maze.getDimensions()
    visited = [[0]*(m + 1) for i in range(n+1)] # BFS 방문 처리를 위한 visited 리스트
    prev = [[(-1,-1)]*(m + 1) for i in range(n+1)] # 경로 역추적을 위한 prev 리스트, 전 위치 좌표를 튜플 형태로 저장한다.
    sx, sy = start_point
    circles = maze.circlePoints()
    gx, gy = circles[0]
    visited[sx][sy] = 1 # 시작 지점 방문 처리
    queue = deque([start_point]) # 큐에 시작 지점 넣어준다.
    while queue:
        x, y = queue.popleft()
        
        if maze.isObjective(x,y) == True: # 도착지에 도착했으면, 도착 지점 저장하고 BFS 종료
            gx, gy = x, y
            break

        nxt = maze.neighborPoints(x, y) # 가능한 인접 후보 노드 expand

        for nx, ny in nxt:
            if visited[nx][ny] == 1: # 방문 되었으면 continue
                continue
            queue.append((nx,ny))
            prev[nx][ny] = (x,y) # 역추적을 위한 이전 노드 저장
            visited[nx][ny] = 1
            
    cx, cy = gx, gy
    while cx != sx or cy != sy: # 경로 역추적을 시작 노드가 나올 때까지 진행
        path.append((cx, cy))
        cx, cy = prev[cx][cy]
    path.append((cx,cy)) # 마지막에 시작 노드 추가 
    path.reverse() # 경로 뒤집어 원상태로 복구
    
    return path
############################################################################

def ids(maze:Maze):
    """
    [Problem 02] 제시된 stage1 맵 세 가지를 IDS를 구현하여 목표지점을 찾는 경로를 return하시오.
    """
    start_point=maze.startPoint()
    path=[]
    ####################### Write Your Code Here ################################
    
    n, m = maze.getDimensions()
    inf = math.inf
    visited = [[inf]*(m+1) for i in range(n+1)] # IDS 방문 처리를 위한 visited 리스트
    prev = [[(-1,-1)]*(m+1) for i in range(n+1)] # 경로 역추적을 위한 prev 리스트, 전 위치 좌표를 튜플 형태로 저장한다.
    circles = maze.circlePoints()
    gx, gy = circles[0]
    sx, sy = start_point
    depth = 0 # IDS 의 깊이 제한을 위한 변수
    def dls(start, limit, k): # depth limited search
        x, y = start
        found = False # 도착점에 도착했는지 여부 

        if maze.isObjective(x, y) == True:
            nonlocal gx # 함수 바깥에 gx
            nonlocal gy # 함수 바깥에 gy
            gx, gy = x, y
            return True

        if k < limit: # depth limit 보다 1 작을 때까지 expand 한다. 
            nxt = maze.neighborPoints(x, y)

            for nx, ny in nxt:
                if visited[nx][ny] <= visited[x][y] + 1: # 현재 거리 + 1 이 이미 저장되어 있는 다음 노드의 거리보다 크거나 같으면 굳이 다시 볼 필요가 없다.
                    continue
                prev[nx][ny] = (x,y) # 경로 역추적을 위한 정보 저장
                visited[nx][ny] = visited[x][y] + 1 # 거리 업데이트
                found = dls((nx,ny), limit, k+1) # 재귀적으로 dls 진행
                if found == True: # 만약 도착 지점에 도착했다면 True가 반환되었을 것이므로 더 이상 볼 필요없이 break
                    break

        if found == True: # 도착점 도착했으므로 return True
            return True
        else:
            return False

    while True: # 도착점을 찾을 때까지 
        for i in range(n+1):
            for j in range(m+1):
                visited[i][j] = inf
        visited[sx][sy] = 0
        found = dls(start_point, depth, 0) # dls
        depth += 1 # depth 1 증가

        if found == True: # 도착점을 찾았을 경우
            cx, cy = gx, gy
            while cx != sx or cy != sy: # 경로 역추적
                path.append((cx,cy))
                cx, cy = prev[cx][cy]
            path.append((cx,cy)) # 시작 지점 추가
            path.reverse() # 경로 뒤집어 원상태 복구
            break

    return path

    #############################################################################

# Manhattan distance
def stage1_heuristic(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) # 맨해튼 거리


def astar(maze:Maze):
    """
    [Problem 03] 제시된 stage1 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.
    (Heuristic Function은 위에서 정의한 stage1_heuristic function(manhattan_dist)을 사용할 것.)
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ################################
    n, m = maze.getDimensions()
    sx, sy = start_point
    circles = maze.circlePoints()
    gx, gy = circles[0]
    inf = math.inf
    dist = [[inf]*(m+1) for i in range(n+1)]
    F = [[inf]*(m+1) for i in range(n+1)]
    prev = [[(-1,-1)]*(m+1) for i in range(n+1)]
    F[sx][sy] = stage1_heuristic((sx,sy),(gx,gy)) # 시작 지점에서부터 도착점까지의 f 값. 
    dist[sx][sy] = 0
    pq = [(F[sx][sy],sx,sy)]
    while pq:
        d_v, x, y = heapq.heappop(pq)
        
        if d_v != F[x][y]: # heappush를 할 때의 거리가 현재 저장되어 있는 F[x][y]와 다르면, F[x][y] 가 작아졌다는 뜻이므로 볼 필요없이 continue
            continue

        if x == gx and y == gy:
            break

        nxt = maze.neighborPoints(x, y)

        for nx, ny in nxt:
            if dist[x][y] + 1 < dist[nx][ny]: # 현재까지의 거리 + 1 이 저장되어 있는 다음 위치까지의 거리보다 작을 때에만 노드 Expand 
                dist[nx][ny] = dist[x][y] + 1
                F[nx][ny] = dist[nx][ny] + stage1_heuristic((nx,ny), (gx,gy)) # f = g + h
                prev[nx][ny] = (x,y) # 경로 역추적을 위한 정보 저장
                heapq.heappush(pq, (F[nx][ny], nx, ny)) # 일반적인 다익스트라와 다르게 F 값을 넣어준다. 

    cx, cy = gx, gy
    while cx != sx or cy != sy: # 경로 역추적
        path.append((cx,cy))
        cx, cy = prev[cx][cy]
    path.append((cx,cy))
    path.reverse()

    return path


    ############################################################################

####################### Write Your Code Here ###############################
class Node(object):
    def __init__(self, past, cur):
        self._past = past # 이전 노드의 위치 튜플
        self._coordinate = cur # 현재 노드의 위치 튜플
        self._leftBit = 0 # 현재 노드의 남은 도착지점들에 대한 비트 정보. 후에 초기화 된다. 해당 인덱스 비트가 1이면 아직 방문하지 않았다는 뜻.
        self._f = 0 # 현재 노드의 f 값
        self._g = 0 # 현재 노드의 g 값
        self._h = 0 # 현재 노드의 휴리스틱 함수 값

    def __lt__(self, other): # 노드를 우선순위 큐에서 F 값을 기준으로 오름차순 정렬하기 때문에 매직 메소드 lt 정의
        return self._f < other._f

def stage2_heuristic(N, Cur, index, to_coord, edge_dist): 
    if Cur._leftBit == 0: # 남은 목표 지점이 없으면 휴리스틱 함수의 반환 값은 0이다. 
        return 0

    left = []
    
    for i in range(N):
        if (1 << i) & Cur._leftBit: # 해당 인덱스의 bit and 연산이 양수면, 아직 방문하지 않은 목표 지점이란 뜻이므로 left에 추가
            left.append(i)

    permu = itertools.permutations(left) # 남은 목표 지점들의 순열을 모두 구한다.

    # stage 2에서 순열을 이용한 이유는 도착지점이 4개로 고정되어 있고, 4! == 24 밖에 되지 않기 때문이다. 
    # 즉, Big-O로 따졌을 때도 무시할 수 있을만한, 그리 크지 않은 상수 값이다. 
    cur_x, cur_y = Cur._coordinate 

    Min = math.inf
    for p in permu:
        total_dist = 0
        for i in range(len(p)):
            if i == 0: # 현재 지점에서부터 순열의 첫 번째 지점까지는 맨해튼 거리로 구해서 더해준다.
                total_dist += (abs(cur_x - to_coord[p[i]][0]) + abs(cur_y - to_coord[p[i]][1]))
            else: # 나머지는 미리 구해둔 도착점과 도착점 사이의 실제 거리를 더해준다. 
                total_dist += edge_dist[p[i]][p[i-1]] 
        
        Min = min(Min, total_dist) # 모든 순열 경우의 수에서 남은 거리가 가장 작은 것을 반환
    return Min

def astar_four_circles(maze: Maze):
    """
    [Problem 05] 제시된 stage3 맵 다섯 가지를 A* Algorithm을 통해 최단 경로를 return하시오.
    (Heuristic Function은 직접 정의 하고, minimum spanning tree를 활용하도록 한다.)
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ###############################
    n, m = maze.getDimensions()
    Start = Node(None, start_point)
    len_circles = len(maze.circlePoints())
    Start._leftBit = (1 << len_circles) - 1 # 시작 노드는 아직 모든 도착점이 방문되지 않았기 때문에 모든 비트가 1로 되어 있다.
    inf = math.inf
    dist = [[inf]*(m+1) for i in range(n+1)]
    edge_dist = [[0]*(len_circles+1) for i in range(len_circles+1)] # 모든 도착지점들 사이의 실제 최단 거리

    coord_to_idx = {} # (x,y) 좌표를 노드 번호로 변환해주는 딕셔너리
    idx_to_coord = {} # 노드 번호를 다시 (x,y) 좌표로  변환해주는 딕셔너리

    dx = [1,0,-1,0]
    dy = [0,1,0,-1]
    for i, p in enumerate(maze.circlePoints()):
        coord_to_idx[p] = i # (x,y) : i
        idx_to_coord[i] = p # i : (x,y)
    
    for cx, cy in maze.circlePoints(): # BFS 알고리즘을 이용해서 미리 도착 지점들 사이의 실제 거리를 구한다.
        # 미로의 각 인접한 칸은 에지가 1인 그래프로 생각해도 되기 때문에 굳이 다익스트라를 통한 최단 거리를 구할 필요가 없다.
        # BFS를 돌면서 특정 지점을 처음 만난 경우에 그때의 길이가 해당 지점과의 최단 거리다.
        for i in range(n+1):
            for j in range(m+1):
                dist[i][j] = math.inf

        # BFS에 대한 설명은 위에서 충분히 하였고, 코드가 거의 동일하다. 
        dist[cx][cy] = 0
        queue = deque([(cx,cy)])
        while queue:
            x, y = queue.popleft()
            
            if (x != cx or y != cy) and maze.isObjective(x,y) == True:
                edge_dist[coord_to_idx[(cx,cy)]][coord_to_idx[(x,y)]] = dist[x][y] # 도착 지점 사이의 거리 edge_dist에 저장
                edge_dist[coord_to_idx[(x,y)]][coord_to_idx[(cx,cy)]] = dist[x][y] # 도착 지점 사이의 거리 edge_dist에 저장

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

# neighborPoints를 쓰지 않은 이유는 전처리 과정은 실제 에이스타 과정에서 노드를 Expand 한 것이 아니기 떄문에 Search States에 포함하지 않았다.
                if nx < 0 or nx >= n or ny < 0 or ny >= m or maze.isWall(nx, ny):
                    continue

                if dist[x][y] + 1 < dist[nx][ny]:
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx,ny))

    ####### 에이스타 알고리즘 ######

    Start._g = 0
    Start._h = stage2_heuristic(len_circles, Start, coord_to_idx, idx_to_coord, edge_dist)
    Start._f = Start._g + Start._h
    pq = [(Start._f,Start)]
    heapq.heapify(pq)
    closed = defaultdict(lambda : math.inf) # 해당 딕셔너리에 키가 존재하지 않으면, math.inf 반환 / (현재 좌표, 현재 노드의 남은 도착 지점 비트) = 현재 노드의 g 값 / graph search
    pre = defaultdict(int) # 경로 저장을 위한 딕셔너리 / (현재 좌표, 현재 노드의 남은 도착 지점 비트) = (이전 노드의 좌표, 이전 노드의 상태)
    goal_list = defaultdict(int) # 도착 지점 확인을 위한 딕셔너리 / (도착점의 좌표) = 1

    for x, y in maze.circlePoints():
        goal_list[(x,y)] = 1

    closed[(start_point, Start._leftBit)] = 0 # 시작 노드 초기화
    while pq:
        d_v, Cur = heapq.heappop(pq)
        x, y = Cur._coordinate

        if d_v != Cur._f: # heappush를 할 때의 거리가 현재 저장되어 있는 F[x][y]와 다르면, F[x][y] 가 작아졌다는 뜻이므로 볼 필요없이 continue
            continue

        if Cur._leftBit == 0: # 모든 도착지를 방문했으면
            cur_x, cur_y = x, y
            state = 0 # 모든 도착지를 방문한 상태는 0 이다. 처음 시작 상태가 모두 1로 되어있는 상태였으니.
            while (cur_x != start_point[0] or cur_y != start_point[1]) or (state != ((1 << len_circles)- 1)): # 시작 좌표, 시작 상태까지 
                path.append((cur_x, cur_y)) # 경로 추가
                ((cur_x, cur_y), state) = pre[((cur_x, cur_y), state)] # 경로 정보를 불러와서 현재로 업데이트
            path.append(start_point)
            path.reverse()
            break

        nxt = maze.neighborPoints(x, y)

        for nx, ny in nxt:
            next_node = Node(Cur, (nx,ny)) # 다음 노드 생성
            next_node._leftBit = Cur._leftBit # 다음 노드의 남은 도착지 비트 정보를 현재 노드의 도착지 비트 정보로 업데이트

# 만약 다음 노드가 도착지이고, 아직 방문하지 않은 도착지인 경우에는 토글하여 해당 비트를 0 으로 바꾸어 주어야 한다. 
            if goal_list[(nx,ny)] == 1 and ((1 << coord_to_idx[(nx,ny)]) & Cur._leftBit): 
                next_node._leftBit ^= (1 << coord_to_idx[(nx,ny)]) # 해당 도착지 방문 표시를 해준다. 해당 노드의 인덱스 번호 비트 1 -> 0 
            
            if closed[(Cur._coordinate, Cur._leftBit)] + 1 < closed[(next_node._coordinate, next_node._leftBit)]: # astar 에서 설명한 것과 동일 현재까지 거리 + 1 < 다음 노드까지의 거리일 때만 expand
                next_node._g = closed[(Cur._coordinate, Cur._leftBit)] + 1 # 출발지부터 다음 노드까지의 거리 = 출발지부터 현재 노드까지의 거리 + 1
                next_node._h = stage2_heuristic(len_circles, next_node, coord_to_idx, idx_to_coord, edge_dist)
                next_node._f = next_node._g + next_node._h
                closed[(next_node._coordinate, next_node._leftBit)] = next_node._g # 방문 표시 업데이트
                pre[(next_node._coordinate, next_node._leftBit)] = (Cur._coordinate, Cur._leftBit) # 경로 역추적을 위한 업데이트
                heapq.heappush(pq, (next_node._f, next_node))
    return path
    ############################################################################
    
    

def stage3_heuristic(N, Cur, index, to_coord, adj):
    if Cur._leftBit == 0:
        return 0

    ## 현재 노드에서 남은 도착지점들까지의 맨해튼 거리를 구하고, 그 중 가장 작은 거리를 ndist에 저장한다.
    cx, cy = Cur._coordinate
    nx, ny = None, None
    Min = math.inf
    for i in range(N):
        if ((1 << i) & Cur._leftBit): # 해당 인덱스가 아직 방문해야 하는 도착지라면
            if abs(to_coord[i][0] - cx) + abs(to_coord[i][1] - cy) < Min: # 현재 지점부터 해당 도착지까지의 맨해튼 거리 구하기
                Min = abs(to_coord[i][0] - cx) + abs(to_coord[i][1] - cy) # 그 중에서 가장 Min 값을 취할 것임.
                nx, ny = to_coord[i] # 해당 도착지의 좌표 저장

    ndist = Min # 선택된 다음 목적지까지의 맨해튼 거리

    # 프림 알고리즘을 사용한 MST 생성 / complete graph 이므로 O(V^2)에 동작
    dist = [math.inf] * (N+1) # 현재 Tree에 속한 노드들과 해당 인덱스까지의 최소 거리
    visited = [0] * (N+1) # 이미 트리에 속한 노드인지
    dist[index[(nx,ny)]] = 0 # 처음 도착지는 0으로 시작
    MST_WEIGHT = 0 # MST 의 총 비용

    for i in range(N):
        now = -1
        min_dist = math.inf
        if (1 << i) & Cur._leftBit == 0: # 이미 방문한 도착지는 MST에 포함시키지 않아야 한다. 
            continue
        for j in range(N):
            if (1 << j) & Cur._leftBit == 0: # 이미 방문한 도착지는 MST에 포함시키지 않아야 한다. 
                continue
            if visited[j] == 0 and dist[j] < min_dist: # 트리에 추가할 다음 노드 후보 정하기
                min_dist = dist[j]
                now = j
        if now < 0:
            continue
        MST_WEIGHT += min_dist # 트리에 해당 노드까지의 거리 추가
        visited[now] = 1 # 방문 표시

        for j in range(N):
            if (1 << j) & Cur._leftBit == 0: # 이미 방문한 도착지는 MST에 포함시키지 않아야 한다. 
                continue
            dist[j] = min(dist[j], adj[now][j]) # 트리의 추가된 노드로부터 dist 갱신

    return ndist + MST_WEIGHT # 반환 값은 (현재 위치 ~ 처음 도착지까지의 맨해튼 거리) + 방문하지 않은 도착지들 간의 MST 비용 

############################################################################

def astar_many_circles(maze: Maze):
    """
    [Problem 05] 제시된 stage3 맵 다섯 가지를 A* Algorithm을 통해 최단 경로를 return하시오.
    (Heuristic Function은 직접 정의 하고, minimum spanning tree를 활용하도록 한다.)
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ###############################
    n, m = maze.getDimensions()
    Start = Node(None, start_point)
    len_circles = len(maze.circlePoints())
    Start._leftBit = (1 << len_circles) - 1
    inf = math.inf
    dist = [[inf]*(m+1) for i in range(n+1)]
    adj = [[-1]* (len_circles+1) for _ in range(len_circles+1)]
    coord_to_idx = {} # (x,y) 좌표를 노드 번호로 변환해주는 딕셔너리
    idx_to_coord = {} # 노드 번호를 다시 (x,y) 좌표로  변환해주는 딕셔너리

    dx = [1,0,-1,0]
    dy = [0,1,0,-1]
    for i, p in enumerate(maze.circlePoints()):
        coord_to_idx[p] = i # (x,y) : i
        idx_to_coord[i] = p # i : (x,y)
    
    # stage 2와 동일한 전처리이므로 자세한 설명은 astar_four_circles 참고.
    for cx, cy in maze.circlePoints(): # BFS 를 이용해서 도착 지점들 사이의 실제 거리를 구한다.
        for i in range(n+1):
            for j in range(m+1):
                dist[i][j] = math.inf

        dist[cx][cy] = 0
        queue = deque([(cx,cy)])
        while queue:
            x, y = queue.popleft()
            
            if (x != cx or y != cy) and maze.isObjective(x,y) == True:
                adj[coord_to_idx[(cx,cy)]][coord_to_idx[(x,y)]] = dist[x][y]
                adj[coord_to_idx[(x,y)]][coord_to_idx[(cx,cy)]] = dist[x][y]

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

                # neighborPoints를 쓰지 않은 이유는 전처리 과정은 실제 에이스타 과정에서 노드를 Expand 한 것이 아니기 떄문에 Search States에 포함하지 않았다.
                if nx < 0 or nx >= n or ny < 0 or ny >= m or maze.isWall(nx, ny): 
                    continue

                if dist[x][y] + 1 < dist[nx][ny]:
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx,ny))

    ####### 에이스타 알고리즘 ######
    # 아래부터는 stage2의 astar_four_circles의 코드와 동일하다. 다른 점은 휴리스틱 함수만 다르기 때문에 자세한 주석은 stage2를 참고.
    Start._g = 0
    Start._h = stage3_heuristic(len_circles, Start, coord_to_idx, idx_to_coord, adj) # 프림 알고리즘 수행을 위해 인접 행렬 adj를 저장한 모습. 값은 두 도착점 간의 실제 최단 거리. 
    Start._f = Start._g + Start._h
    pq = [(Start._f,Start)]
    heapq.heapify(pq)

    closed = defaultdict(lambda : math.inf)
    pre = defaultdict(int)
    goal_list = defaultdict(int)
    for x, y in maze.circlePoints():
        goal_list[(x,y)] = 1

    closed[(start_point, Start._leftBit)] = 0
    while pq:
        d_v, Cur = heapq.heappop(pq)
        x, y = Cur._coordinate
        
        if d_v != Cur._f: # heappush를 할 때의 거리가 현재 저장되어 있는 F[x][y]와 다르면, F[x][y] 가 작아졌다는 뜻이므로 볼 필요없이 continue
            continue
    
        if Cur._leftBit == 0:
            cur_x, cur_y, cur_g = x, y, Cur._g
            state = 0 # 모든 도착지를 방문한 상태는 0 이다. 처음 시작 상태가 모두 1로 되어있는 상태였으니.
            while (cur_x != start_point[0] or cur_y != start_point[1]) or (state != ((1 << len_circles)- 1)): # 시작 좌표, 시작 상태까지 
                path.append((cur_x, cur_y))
                ((cur_x, cur_y), state) = pre[((cur_x, cur_y), state)]
                cur_g -= 1
            path.append(start_point)
            path.reverse()
            break

        nxt = maze.neighborPoints(x, y)

        for nx, ny in nxt:
            next_node = Node(Cur, (nx,ny))
            next_node._leftBit = Cur._leftBit

# 만약 다음 노드가 도착지이고, 아직 방문하지 않은 도착지인 경우에는 토글하여 해당 비트를 0 으로 바꾸어 주어야 한다. 
            if goal_list[(nx,ny)] == 1 and ((1 << coord_to_idx[(nx,ny)]) & Cur._leftBit): 
                next_node._leftBit ^= (1 << coord_to_idx[(nx,ny)]) # 해당 도착지 방문 표시를 해준다. 해당 노드의 인덱스 번호 비트 1 -> 0 
            
            if closed[(Cur._coordinate, Cur._leftBit)] + 1 < closed[(next_node._coordinate, next_node._leftBit)]:
                next_node._g = closed[(Cur._coordinate, Cur._leftBit)] + 1
                next_node._h = stage3_heuristic(len_circles, next_node, coord_to_idx, idx_to_coord, adj) # 이 부분만 stage2와 다른 부분이다. 
                next_node._f = next_node._g + next_node._h
                closed[(next_node._coordinate, next_node._leftBit)] = next_node._g
                pre[(next_node._coordinate, next_node._leftBit)] = (Cur._coordinate, Cur._leftBit)
                heapq.heappush(pq, (next_node._f, next_node))

    return path
    ############################################################################