import random
import uuid

from enum import Enum
from typing import Self, Optional


MAP_HEIGTH = 128
MAP_WIDTH = 128
PRIMORDIAL_BROTH_DENSITY = 0.5
NUMBER_OF_ROUNDS = 1000
MAX_LIFE_EXPECTANCY = 100


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
    def get_from_raw_results(cls, results: tuple[
        float, # north
        float, # north-east
        float, # east
        float, # south-east
        float, # south
        float, # south-west
        float, # west
        float, # north-west
        float, # stay and defend
        float, # walk
        float, # love
        float # fight
    ]) -> Self:
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

        if random.uniform(0, 1) > (1 - PRIMORDIAL_BROTH_DENSITY):
            self.creature = Creature.born_from_primordial_broth()
        else:
            self.creature = None
    
    def get_empty_cell_nearby(self, field: list[list[Self]]) -> Optional[Self]:
        for direction in Direction:
            coordinates = self.coordinates.shift(direction=direction)
            if coordinates:
                neighbour = Cell.get(field=field, coordinates=coordinates)
                if not neighbour.creature:
                    return neighbour

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
            self.genes = [0.0] # TODO
        else:
            self.genes = [0.0] # TODO
    
    @classmethod
    def born_from_primordial_broth(cls) -> Self:
        return cls(father=None, mother=None)
    
    @classmethod
    def born(cls, father: Self, mother: Self) -> Self:
        return cls(father=father, mother=mother)
    
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
        
        results = self.decide(
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
        return random.uniform(0, 1) # TODO

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

    def decide(
        self,
        north_neighbour: float,
        north_neighbour_satiety: float,
        north_neighbour_age: float,
        north_neighbour_similarity: float,
        north_east_neighbour: float,
        north_east_neighbour_satiety: float,
        north_east_neighbour_age: float,
        north_east_neighbour_similarity: float,
        east_neighbour: float,
        east_neighbour_satiety: float,
        east_neighbour_age: float,
        east_neighbour_similarity: float,
        south_east_neighbour: float,
        south_east_neighbour_satiety: float,
        south_east_neighbour_age: float,
        south_east_neighbour_similarity: float,
        south_neighbour: float,
        south_neighbour_satiety: float,
        south_neighbour_age: float,
        south_neighbour_similarity: float,
        south_west_neighbour: float,
        south_west_neighbour_satiety: float,
        south_west_neighbour_age: float,
        south_west_neighbour_similarity: float,
        west_neighbour: float,
        west_neighbour_satiety: float,
        west_neighbour_age: float,
        west_neighbour_similarity: float,
        north_west_neighbour: float,
        north_west_neighbour_satiety: float,
        north_west_neighbour_age: float,
        north_west_neighbour_similarity: float,
        satiety: float,
        age: float
    ) -> tuple[
        float, # north
        float, # north-east
        float, # east
        float, # south-east
        float, # south
        float, # south-west
        float, # west
        float, # north-west
        float, # stay and defend
        float, # walk
        float, # love
        float # fight
    ]:
        return (
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1)
        ) # TODO
    
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

    def __init__(self):
        self.field = [[]]
        self.plans = []

        for y in range(0, MAP_HEIGTH):
            self.field.append([])

            for x in range(0, MAP_WIDTH):
                coordinates = Coordinates(x=x, y=y)
                self.field[y].append(Cell(coordinates))
    
    def run(self):
        for round in range(0, NUMBER_OF_ROUNDS):
            print(f"round = {round}")
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
            print(f"lived creatures = {lived_creatures}")

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
        return None
    
    def resolve_plan(self, plan: Plan):
        if plan.cell.creature:
            coordinates = plan.target_coordinates
            if coordinates:
                target = Cell.get(field=self.field, coordinates=coordinates)
                target_decision = next((p.decision for p in self.plans if p.cell == target), None)
                if target_decision:
                    self.resolve_plan_with_target(plan=plan, target=target, target_decision=target_decision)
                    return
            self.resolve_plan_without_target(plan=plan)

    def resolve_plan_with_target(self, plan: Plan, target: Cell, target_decision: Decision):
        match plan.decision.action:
            case Action.Walk:
                if not target.creature:
                    target.creature = plan.cell.creature
                    plan.cell.creature = None
            case Action.Love:
                if target.creature and plan.cell.creature.satiety and target.creature.satiety:
                    father = plan.cell.creature
                    mother = target.creature
                    child = Creature.born(father=father, mother=mother)
                    print(f"creature was born {child}")
                    father.starve()
                    mother.starve()
                    empty_cell = plan.cell.get_empty_cell_nearby(field=self.field)
                    if empty_cell:
                        empty_cell.creature = child
            case Action.Fight:
                if target.creature and not target_decision.defended:
                    print(f"creature was killed {target.creature}")
                    target.creature = None
                    plan.cell.creature.eat()
            case Action.Die:
                print(f"creature was died (with_target) {plan.cell.creature}")
                plan.cell.creature = None

    def resolve_plan_without_target(self, plan: Plan):
        if plan.decision.action == Action.Die:
            print(f"creature was died (without_target) {plan.cell.creature}")
            plan.cell.creature = None


map = Map()
map.run()
