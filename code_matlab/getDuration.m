function durations = getDuration(triggers)
    
    T = readtable('trigger_list.xlsx');
    durations = zeros(1, length(triggers));
    for ti = 1:length(triggers)
        if iscell(triggers)
            trigger = triggers{ti};
            trigger = str2num(trigger(2:end));
        else
            trigger = triggers(ti);
        end
        tp = T(T.trigger ==trigger, 'scaling');
        tp = table2array(tp);
        if iscell(tp)
            durations(ti) = str2num(tp{1})/100*0.6;
        else
            durations(ti) = tp/100*0.6;
        end
    end
end