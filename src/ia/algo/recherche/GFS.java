package ia.algo.recherche;

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;

import ia.framework.common.State;
import ia.framework.common.Action;
import ia.framework.common.ArgParse;
import ia.framework.recherche.TreeSearch;
import ia.framework.recherche.SearchProblem;
import ia.framework.recherche.SearchNode;
import ia.framework.recherche.HasHeuristic;

/**
 * Algorithme de recherche gloutonne (Greedy First Search)
 * Explore les nœuds par ordre croissant d'heuristique
 */
public class GFS extends TreeSearch {

    public GFS(SearchProblem prob, State initial_state){
        super(prob, initial_state);
        // File de priorité ordonnée par heuristique croissante
        this.frontier = new PriorityQueue<SearchNode>(
            Comparator.comparingDouble(node -> {
                State s = node.getState();
                if (s instanceof HasHeuristic) {
                    return ((HasHeuristic) s).getHeuristic();
                }
                return 0.0;
            })
        );
    }

    public boolean solve() {

        // 1 Créer un nœud correspondant à l'état initial
        SearchNode root_node = SearchNode.makeRootSearchNode(initial_state);

        // 2 Initialiser la frontière avec ce nœud
        this.frontier.clear();
        this.frontier.add(root_node);

        // 3 Initialiser l'ensemble des états visités à vide
        this.explored.clear();

        // 4 Tant que la frontière n'est pas vide
        while (!this.frontier.isEmpty()) {

            // 5 Retirer le nœud d'heuristique minimale
            SearchNode cur_node = this.frontier.poll();

            // 6 Si le nœud contient un état but
            State cur_state = cur_node.getState();
            if (problem.isGoalState(cur_state)) {
                this.end_node = cur_node;
                return true;
            }

            // 9 Ajouter son état à l'ensemble des états visités
            this.explored.add(cur_state);

            // 10 Étendre les enfants du nœud
            ArrayList<Action> actions = problem.getActions(cur_state);
            if (ArgParse.DEBUG)
                System.out.print(cur_state+" ("+actions.size()+" actions) -> {");

            // 11 Pour chaque nœud enfant
            for (Action a : actions) {

                SearchNode child_node = SearchNode.makeChildSearchNode(problem, cur_node, a);
                State child_state = child_node.getState();

                if (ArgParse.DEBUG)
                    System.out.print("("+a+", "+child_state+")");

                // 12 S'il n'est pas dans la frontière et si son état n'a pas été visité
                if (!frontier.contains(child_node) && !explored.contains(child_state)) {

                    // 13 L'insérer dans la frontière
                    frontier.add(child_node);

                    if (ArgParse.DEBUG)
                        System.out.print("[A] ");

                } else {
                    if (ArgParse.DEBUG)
                        System.out.print("[I] ");
                }
            }
            if (actions.size() > 0 && ArgParse.DEBUG)
                System.out.println("}");
        }

        if (ArgParse.DEBUG)
            System.out.println();
        return false;
    }
}
