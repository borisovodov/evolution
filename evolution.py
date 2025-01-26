import random
import uuid
from enum import Enum
from typing import Self, Optional

from genes import Genes

MAP_HEIGTH = 128
MAP_WIDTH = 128
PRIMORDIAL_BROTH_DENSITY = 0.001
PRIMORDIAL_BROTH_DURATION = 1000
NUMBER_OF_ROUNDS = 5000
MAX_LIFE_EXPECTANCY = 1000


class Direction(Enum):
    North = 0
    NorthEast = 1
    East = 2
    SouthEast = 3
    South = 4
    SouthWest = 5
    West = 6
    NorthWest = 7


class Action(Enum):
    StayAndDefend= 0
    Walk = 1
    Love = 2
    Fight = 3
    Die = 4


class Decision:
    def __init__(self, direction: Direction, action: Action):
        self.direction = direction
        self.action = action
    
    @classmethod
    def get(cls, direction: Direction, action: Action) -> Self:
        return cls(direction=direction, action=action)
    
    @classmethod
    def get_from_raw_results(cls, results: list) -> Self:
        index_max_direction = max(range(0, 8), key=results.__getitem__)
        index_max_action = max(range(8, 12), key=results.__getitem__)

        match index_max_direction:
            case 0:
                direction = Direction.North
            case 1:
                direction = Direction.NorthEast
            case 2:
                direction = Direction.East
            case 3:
                direction = Direction.SouthEast
            case 4:
                direction = Direction.South
            case 5:
                direction = Direction.SouthWest
            case 6:
                direction = Direction.West
            case 7:
                direction = Direction.NorthWest
        
        match index_max_action:
            case 8:
                action = Action.StayAndDefend
            case 9:
                action = Action.Walk
            case 10:
                action = Action.Love
            case 11:
                action = Action.Fight
        return cls(direction=direction, action=action)
    
    @property
    def defended(self) -> bool:
        return self.action == Action.StayAndDefend
    
    def __str__(self):
        return f"Decision(action={self.action}, direction={self.direction})"


class Coordinates:
    def __init__(self, x, y):
        if x < MAP_WIDTH and y < MAP_HEIGTH and x >= 0 and y >= 0:
            self.x = x
            self.y = y
        else:
            raise ValueError("Coordinates out of bounds")

    def shift(self, direction: Direction) -> Optional[Self]:
        match direction:
            case Direction.North:
                if self.y < (MAP_HEIGTH - 1):
                    return Coordinates(self.x, self.y + 1)
                return None
            case Direction.NorthEast:
                if self.x < (MAP_WIDTH - 1) and self.y < (MAP_HEIGTH - 1):
                    return Coordinates(self.x + 1, self.y + 1)
                return None
            case Direction.East:
                if self.x < (MAP_WIDTH - 1):
                    return Coordinates(self.x + 1, self.y)
                return None
            case Direction.SouthEast:
                if self.x < (MAP_WIDTH - 1) and self.y > 0:
                    return Coordinates(self.x + 1, self.y - 1)
                return None
            case Direction.South:
                if self.y > 0:
                    return Coordinates(self.x, self.y - 1)
                return None
            case Direction.SouthWest:
                if self.x > 0 and self.y > 0:
                    return Coordinates(self.x - 1, self.y - 1)
                return None
            case Direction.West:
                if self.x > 0:
                    return Coordinates(self.x - 1, self.y)
                return None
            case Direction.NorthWest:
                if self.x > 0 and self.y < (MAP_HEIGTH - 1):
                    return Coordinates(self.x - 1, self.y + 1)
                return None
            
    def __str__(self):
        return f"Coordinates(x={self.x}, y={self.y})"


class Cell:
    def __init__(self, coordinates: Coordinates):
        self.coordinates = coordinates
        self.creature = None
    
    def get_empty_cell_nearby(self, field: list[list[Self]]) -> Optional[Self]:
        for direction in Direction:
            coordinates = self.coordinates.shift(direction=direction)
            if coordinates:
                neighbour = Cell.get(field=field, coordinates=coordinates)
                if not neighbour.creature:
                    return neighbour
        return None

    @staticmethod
    def get(field: list[list[Self]], coordinates: Coordinates) -> Self:
        return field[coordinates.y][coordinates.x]
    
    def __str__(self):
        return f"Cell(coordinates={self.coordinates}, creature={self.creature})"


class Creature:
    def __init__(self, father: Optional[Self], mother: Optional[Self]):
        self.id = uuid.uuid4()
        self.age = 0
        self.satiety = False

        if father and mother:
            self.genes = Genes(father_genes=father.genes, mother_genes=mother.genes)
        else:
            self.genes = Genes(None, None)
    
    def plan_move(self, neighbour_cells: dict[Direction:Optional[Cell]]) -> Decision:
        if self.age >= MAX_LIFE_EXPECTANCY:
            return Decision.get(Direction.North, Action.Die)
        else:
            self.age += 1
        
        north_neighbour, north_neighbour_satiety, north_neighbour_age, north_neighbour_similarity = self.handle_direction(direction=Direction.North, neighbour_cells=neighbour_cells)
        north_east_neighbour, north_east_neighbour_satiety, north_east_neighbour_age, north_east_neighbour_similarity = self.handle_direction(direction=Direction.NorthEast, neighbour_cells=neighbour_cells)
        east_neighbour, east_neighbour_satiety, east_neighbour_age, east_neighbour_similarity = self.handle_direction(direction=Direction.East, neighbour_cells=neighbour_cells)
        south_east_neighbour, south_east_neighbour_satiety, south_east_neighbour_age, south_east_neighbour_similarity = self.handle_direction(direction=Direction.SouthEast, neighbour_cells=neighbour_cells)
        south_neighbour, south_neighbour_satiety, south_neighbour_age, south_neighbour_similarity = self.handle_direction(direction=Direction.South, neighbour_cells=neighbour_cells)
        south_west_neighbour, south_west_neighbour_satiety, south_west_neighbour_age, south_west_neighbour_similarity = self.handle_direction(direction=Direction.SouthWest, neighbour_cells=neighbour_cells)
        west_neighbour, west_neighbour_satiety, west_neighbour_age, west_neighbour_similarity = self.handle_direction(direction=Direction.West, neighbour_cells=neighbour_cells)
        north_west_neighbour, north_west_neighbour_satiety, north_west_neighbour_age, north_west_neighbour_similarity = self.handle_direction(direction=Direction.NorthWest, neighbour_cells=neighbour_cells)
        
        satiety = 1.0 if self.satiety else 0.0
        age = self.age / MAX_LIFE_EXPECTANCY
        
        results = self.genes.decide(
            north_neighbour,
            north_neighbour_satiety,
            north_neighbour_age,
            north_neighbour_similarity,
            north_east_neighbour,
            north_east_neighbour_satiety,
            north_east_neighbour_age,
            north_east_neighbour_similarity,
            east_neighbour,
            east_neighbour_satiety,
            east_neighbour_age,
            east_neighbour_similarity,
            south_east_neighbour,
            south_east_neighbour_satiety,
            south_east_neighbour_age,
            south_east_neighbour_similarity,
            south_neighbour,
            south_neighbour_satiety,
            south_neighbour_age,
            south_neighbour_similarity,
            south_west_neighbour,
            south_west_neighbour_satiety,
            south_west_neighbour_age,
            south_west_neighbour_similarity,
            west_neighbour,
            west_neighbour_satiety,
            west_neighbour_age,
            west_neighbour_similarity,
            north_west_neighbour,
            north_west_neighbour_satiety,
            north_west_neighbour_age,
            north_west_neighbour_similarity,
            satiety,
            age
        )

        return Decision.get_from_raw_results(results)

    def eat(self):
        self.satiety = True

    def starve(self):
        self.satiety = False

    def similarity(self, creature: Self) -> float:
        return 0.0 # TODO

    def handle_direction(
        self,
        direction: Direction,
        neighbour_cells: dict[Direction:Optional[Cell]]
    ) -> tuple[
        float, # is cell full
        float, # satiety
        float, # age
        float # similarity
    ]:
        if neighbour_cells.get(direction):
            if neighbour_cells.get(direction).creature:
                neighbour_creature = neighbour_cells.get(direction).creature
                satiety = 1.0 if neighbour_creature.satiety else 0.0
                age = neighbour_creature.age / MAX_LIFE_EXPECTANCY
                return (1.0, satiety, age, self.similarity(creature=neighbour_creature))
        return (0.0, 0.0, 0.0, 0.0)
    
    def __str__(self):
        return f"Creature(id={self.id}, age={self.age}, satiety={self.satiety})"


class Plan:
    def __init__(self, cell: Cell, decision: Decision):
        self.cell = cell
        self.decision = decision

    @property
    def target_coordinates(self) -> Optional[Coordinates]:
        return self.cell.coordinates.shift(direction=self.decision.direction)
    
    def __str__(self):
        return f"Plan(cell={self.cell}, decision={self.decision})"


class Map:
    field: list[list[Cell]]
    plans: list[Plan]
    round: int
    log: list[str]

    def __init__(self):
        self.field = [[]]
        self.plans = []
        self.round = 0
        self.log = []

        for y in range(0, MAP_HEIGTH):
            self.field.append([])

            for x in range(0, MAP_WIDTH):
                coordinates = Coordinates(x=x, y=y)
                self.field[y].append(Cell(coordinates))
    
    def run(self):
        for round in range(0, NUMBER_OF_ROUNDS):
            self.round = round
            print(f"round = {self.round}")
            self.log.append(f"round = {self.round}")
            self.plans = []
            lived_creatures = 0

            for row in self.field:
                for cell in row:
                    plan = self.handle_plan_move(cell=cell)
                    if plan:
                        self.plans.append(plan)
                        lived_creatures += 1
            for plan in self.plans:
                self.resolve_plan(plan=plan)
            self.log.append(f"lived creatures = {lived_creatures}")
            print(f"lived creatures = {lived_creatures}")
        with open("log.txt", "w") as file:
            for line in self.log:
                file.write(f"{line}\n")

    def handle_plan_move(self, cell: Cell) -> Optional[Plan]:
        if cell.creature:
            neighbour_cells = {}
            for direction in Direction:
                coordinates = cell.coordinates.shift(direction=direction)
                if coordinates:
                    neighbour_cell = Cell.get(field=self.field, coordinates=coordinates)
                    neighbour_cells[direction] = neighbour_cell
                else:
                    neighbour_cells[direction] = None
            decision = cell.creature.plan_move(neighbour_cells=neighbour_cells)
            return Plan(cell=cell, decision=decision)
        else:
            if random.uniform(0, 1) < self.born_from_primordial_broth_probability:
                cell.creature = Creature(None, None)
                self.log.append(f"creature was born from primordial broth {cell.creature}")
            return None
    
    @property
    def born_from_primordial_broth_probability(self) -> float:
        return PRIMORDIAL_BROTH_DENSITY * (1 - (self.round / PRIMORDIAL_BROTH_DURATION))
    
    def resolve_plan(self, plan: Plan):
        if plan.cell.creature:
            coordinates = plan.target_coordinates
            if coordinates:
                target = Cell.get(field=self.field, coordinates=coordinates)
                target_decision = next((p.decision for p in self.plans if p.cell == target), None)
                self.resolve_plan_with_target(plan=plan, target=target, target_decision=target_decision)
                return
            self.resolve_plan_without_target(plan=plan)

    def resolve_plan_with_target(self, plan: Plan, target: Cell, target_decision: Optional[Decision]):
        match plan.decision.action:
            case Action.StayAndDefend:
                self.log.append(f"creature stay and defended {plan.cell.creature}")
            case Action.Walk:
                if not target.creature:
                    target.creature = plan.cell.creature
                    plan.cell.creature = None
                    self.log.append(f"creature move {plan.cell.creature}") # creature move None !!!!!!!!
                else:
                    self.log.append(f"creature want to move, but too crowd {plan.cell.creature}")
            case Action.Love:
                #if target.creature and plan.cell.creature.satiety and target.creature.satiety:
                if target.creature and plan.cell.creature.satiety:
                    father = plan.cell.creature
                    mother = target.creature
                    father.starve()
                    mother.starve()
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    else:
                        self.log.append(f"creature was not born, because too crowd")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = Creature(father=father, mother=mother)
                        self.log.append(f"creature was born {empty_cell.creature} from father {plan.cell.creature} and mother {target.creature}")
                else:
                    self.log.append(f"creature want to love, but there is no creature in target cell or creatures are not satiated {plan.cell.creature}")
            case Action.Fight:
                if target.creature and target_decision and not target_decision.defended:
                    target.creature = None
                    plan.cell.creature.eat()
                    self.log.append(f"creature kill other creature {plan.cell.creature}")
                else:
                    self.log.append(f"creature want to fight, but target defended or there is no creature in target {plan.cell.creature}")
            case Action.Die:
                plan.cell.creature = None
                self.log.append(f"creature die from old age {plan.cell.creature}")

    def resolve_plan_without_target(self, plan: Plan):
        match plan.decision.action:
            case Action.StayAndDefend:
                self.log.append(f"creature stay and defended (without target cell) {plan.cell.creature}")
            case Action.Walk:
                self.log.append(f"creature want to move, but there is no target cell {plan.cell.creature}")
            case Action.Love:
                self.log.append(f"creature want to love, but there is no target cell {plan.cell.creature}")
            case Action.Fight:
                self.log.append(f"creature want to fight, but there is no target cell {plan.cell.creature}")
            case Action.Die:
                plan.cell.creature = None
                self.log.append(f"creature die from old age without target {plan.cell.creature}")


map = Map()
map.run()
