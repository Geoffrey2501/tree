

import ia.algo.recherche.*;
import ia.framework.common.State;
import ia.framework.recherche.SearchNode;
import ia.framework.recherche.TreeSearch;
import ia.problemes.Dummy;

import java.util.ArrayList;
import java.util.List;

public class MainBenchmark {

    static class Result {
        String algo;
        int n, k;
        long seed;
        boolean success;
        long timeMs;
        int explored;
        int depth;
        double cost;

        @Override
        public String toString() {
            return algo + ";" + n + ";" + k + ";" + seed + ";" +
                    success + ";" + timeMs + ";" +
                    explored + ";" + depth + ";" + cost;
        }
    }

    public static void main(String[] args) {

        int[] Ns = {100, 1000, 10000};
        int[] Ks = {2, 4, 6};
        long seed = 42L;

        List<Result> results = new ArrayList<>();

        System.out.println("algo;n;k;seed;success;time_ms;explored;depth;cost");

        for (int n : Ns) {
            for (int k : Ks) {

                Dummy problem = new Dummy(n, k, seed);
                State init = Dummy.initialState(); // <-- correction ici

                for (TreeSearch algo : List.of(new BFS(problem, init), new DFS(problem, init), new UCS(problem, init), new AStar(problem, init),new GFS(problem, init))) {

                    long start = System.currentTimeMillis();
                    boolean success = algo.solve();
                    long time = System.currentTimeMillis() - start;

                    Result r = new Result();
                    r.algo = algo.getClass().getSimpleName();
                    r.n = n; r.k = k; r.seed = seed;
                    r.success = success; r.timeMs = time;
                    r.explored = algo.getVisited().size();

                    if (success) {
                        SearchNode goal = algo.getEndNode();
                        r.depth = goal.getDepth();
                        r.cost = goal.getCost();
                    } else {
                        r.depth = -1;
                        r.cost = -1;
                    }

                    results.add(r);
                    System.out.println(r);
                }
            }
        }
    }
}
