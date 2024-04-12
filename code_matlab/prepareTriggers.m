%%
% format triggers:
% turn numerical triggers into strings that match the recoded file in Brain Vision 
% input is an arrary/cell and output is a cell
function tp = prepareTriggers(triggers)
    
    tp = string(triggers);
    tp = pad(tp, 3, 'left');
    tp = pad(tp, 4, 'left', 'S');
    tp = cellstr(tp);
        
end