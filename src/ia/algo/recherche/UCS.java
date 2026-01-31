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

/**
 * Algorithme de coût uniforme (Uniform Cost Search)
 * Explore les nœuds par ordre croissant de coût de chemin
 */
public class UCS extends TreeSearch {

    public UCS(SearchProblem prob, State initial_state){
        super(prob, initial_state);
        // File de priorité ordonnée par coût croissant
        this.frontier = new PriorityQueue<SearchNode>(
            Comparator.comparingDouble(SearchNode::getCost)
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
        this.visited.clear();

        // 4 Tant que la frontière n'est pas vide
        while (!this.frontier.isEmpty()) {

            // 5 Retirer le nœud de coût minimal
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

                // 12 S'il n'a pas été visité
                if (!explored.contains(child_state)) {

                    // Vérifier si un meilleur chemin existe déjà
                    SearchNode existing = visited.get(child_state);

                    if (existing == null) {
                        // Nouveau nœud
                        frontier.add(child_node);
                        visited.put(child_state, child_node);

                        if (ArgParse.DEBUG)
                            System.out.print("[A] ");
                    } else if (child_node.getCost() < existing.getCost()) {
                        // Meilleur chemin trouvé
                        frontier.remove(existing);
                        frontier.add(child_node);
                        visited.put(child_state, child_node);

                        if (ArgParse.DEBUG)
                            System.out.print("[U] ");
                    } else {
                        if (ArgParse.DEBUG)
                            System.out.print("[I] ");
                    }
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
